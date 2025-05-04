import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Self-attention mechanism."""
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, Key, Value transformations
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        # Dropout and final transformation
        self.dropout = nn.Dropout(dropout_prob)
        self.output_layer = nn.Linear(hidden_size, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size)
    
    def transpose_for_scores(self, x):
        """Reshape from [batch_size, seq_len, hidden_size] to [batch_size, num_heads, seq_len, head_size]."""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # Project inputs to queries, keys, and values
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Take the dot product to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply the attention mask (if provided)
        if attention_mask is not None:
            # Add large negative values to masked positions
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_scores = attention_scores + extended_attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        
        # This is actually dropping out entire tokens to attend to, which we found
        # works better than dropping out individual attention probabilities
        attention_probs = self.dropout(attention_probs)
        
        # Calculate the context vector
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # Apply output transformation
        output = self.output_layer(context_layer)
        
        # Apply layer normalization (with residual connection)
        output = self.layernorm(output + hidden_states)
        
        return output


class CDP_JIT_SDP(nn.Module):
    def __init__(self, args):
        super(CDP_JIT_SDP, self).__init__()
        self.args = args
        
        V_msg = args.vocab_msg
        V_code = args.vocab_code
        Dim = args.embedding_dim
        Class = args.class_num
        
        Ci = 1  # input of convolutional layer
        Co = args.num_filters  # output of convolutional layer
        Ks = args.filter_sizes  # kernel sizes
        
        # Number of attention heads and dropout rate
        self.num_attention_heads = args.num_attention_heads if hasattr(args, 'num_attention_heads') else 8
        self.dropout_prob = args.dropout_keep_prob
        
        # CNN-2D for commit message
        self.embed_msg = nn.Embedding(V_msg, Dim)
        self.convs_msg = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        
        # Self-attention for message features
        self.msg_attention = SelfAttention(
            hidden_size=Co * len(Ks),
            num_attention_heads=self.num_attention_heads,
            dropout_prob=self.dropout_prob
        )
        
        # CNN-2D for commit code
        self.embed_code = nn.Embedding(V_code, Dim)
        self.convs_code_line = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Dim)) for K in Ks])
        self.convs_code_file = nn.ModuleList([nn.Conv2d(Ci, Co, (K, Co * len(Ks))) for K in Ks])
        
        # Self-attention for code features
        self.code_attention = SelfAttention(
            hidden_size=Co * len(Ks),
            num_attention_heads=self.num_attention_heads,
            dropout_prob=self.dropout_prob
        )
        
        # Cross-attention between message and code
        self.cross_attention = SelfAttention(
            hidden_size=Co * len(Ks),
            num_attention_heads=self.num_attention_heads,
            dropout_prob=self.dropout_prob
        )
        
        # Final classification layers
        self.dropout = nn.Dropout(args.dropout_keep_prob)
        
        # Increase hidden units for better representation
        hidden_size = 2 * Co * len(Ks)
        self.fc1 = nn.Linear(hidden_size, args.hidden_units)
        self.bn1 = nn.BatchNorm1d(args.hidden_units)
        
        # Add a second hidden layer for better feature extraction
        self.fc2 = nn.Linear(args.hidden_units, args.hidden_units // 2)
        self.bn2 = nn.BatchNorm1d(args.hidden_units // 2)
        
        # Output layer
        self.fc_out = nn.Linear(args.hidden_units // 2, Class)
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights with better defaults
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.01)  # Small positive bias
    
    def forward_msg(self, x, convs):
        # Add sequence dimension if not present
        x = x.unsqueeze(1) if x.dim() == 3 else x  # (N, Ci, W, D)
        
        # Apply CNN to extract features
        x = [F.relu(conv(x)).squeeze(3) for conv in convs]  # [(N, Co, W), ...]*len(Ks)
        
        # Apply max-pooling over time
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        
        # Concatenate filter outputs
        x = torch.cat(x, 1)  # (N, Co * len(Ks))
        
        return x
    
    def forward_code_with_attention(self, x, convs_line, convs_hunks):
        n_batch, n_file = x.shape[0], x.shape[1]
        
        # Process each line with CNN
        x_reshaped = x.reshape(n_batch * n_file, x.shape[2], x.shape[3])
        x_line_features = self.forward_msg(x_reshaped, convs_line)
        
        # Reshape back to get features per file
        x_line_features = x_line_features.reshape(n_batch, n_file, -1)
        
        # Apply self-attention over lines in each file
        x_line_attention = self.code_attention(x_line_features)
        
        # Process each file with CNN (using attended features)
        x_file_features = self.forward_msg(x_line_attention.unsqueeze(1), convs_hunks)
        
        return x_file_features
    
    def forward(self, msg, code):
        # Process commit message
        x_msg = self.embed_msg(msg)
        x_msg_cnn = self.forward_msg(x_msg, self.convs_msg)
        
        # Reshape for attention
        x_msg_attention = x_msg_cnn.unsqueeze(1)  # Add sequence dimension
        
        # Apply self-attention to message features
        x_msg_attended = self.msg_attention(x_msg_attention).squeeze(1)
        
        # Process code changes with attention
        x_code = self.embed_code(code)
        x_code_attended = self.forward_code_with_attention(
            x_code, self.convs_code_line, self.convs_code_file
        )
        
        # Combine message and code features
        x_commit = torch.cat((x_msg_attended, x_code_attended), 1)
        
        # Apply dropout
        x_commit = self.dropout(x_commit)
        
        # First hidden layer with batch normalization
        out = self.fc1(x_commit)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second hidden layer with batch normalization
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Output layer without sigmoid (for use with BCEWithLogitsLoss)
        out = self.fc_out(out)
        
        return out.squeeze(1)
    
    def get_embeddings(self, msg, code):
        """
        Extract embeddings for concept drift detection.
        
        Returns:
            Combined embeddings from the last hidden layer
        """
        # Process commit message
        x_msg = self.embed_msg(msg)
        x_msg_cnn = self.forward_msg(x_msg, self.convs_msg)
        x_msg_attention = x_msg_cnn.unsqueeze(1)
        x_msg_attended = self.msg_attention(x_msg_attention).squeeze(1)
        
        # Process code changes with attention
        x_code = self.embed_code(code)
        x_code_attended = self.forward_code_with_attention(
            x_code, self.convs_code_line, self.convs_code_file
        )
        
        # Combine message and code features
        x_commit = torch.cat((x_msg_attended, x_code_attended), 1)
        
        # First hidden layer features (for drift detection)
        out = self.fc1(x_commit)
        out = self.bn1(out)
        out = F.relu(out)
        
        return out 