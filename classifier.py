import torch as tc


class Conv1dTextClassifier(tc.nn.Module):
    def __init__(self, num_classes):
        super(Conv1dTextClassifier, self).__init__()
        self.emb_dim = 300
        self.num_filters = 100
        self.conv_kernel_sizes = [3, 4, 5]
        self.convs = tc.nn.ModuleList()
        for k in self.conv_kernel_sizes:
            conv_stack = tc.nn.Sequential(
                tc.nn.Conv1d(self.emb_dim, self.num_filters, kernel_size=(k,), stride=(1,), dilation=(1,)),
                tc.nn.ReLU()
            )
            self.convs.append(conv_stack)
        self.dropout = tc.nn.Dropout(p=0.5)
        self.fc = tc.nn.Linear(len(self.conv_kernel_sizes) * self.num_filters, num_classes)

    def forward(self, x):
        features = []
        for l in range(len(self.conv_kernel_sizes)):
            c = self.convs[l](x)
            m, _ = tc.max(c, dim=-1) # [B, C, T] -> [B, C].
            features.append(m)

        vec = tc.cat(tuple(features), dim=-1)
        d = self.dropout(vec)
        logits = self.fc(d)
        return logits


