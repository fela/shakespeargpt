import torch
import datasets


all_data = datasets.load_dataset(name='tiny_shakespear')['train']

print(all_data)

chars = ''.join(set('<placeholder>'))
input_dim = len(chars)
char_to_i = {c: i for i, c in enumerate(chars)}
i_to_char = {i: c for i, c in enumerate(chars)}
context_size = 8
print_loss_every_n_steps = 100


class Net(torch.nn.Module):
    def __init__(self):
        self.logits = torch.nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x):
        # one hot encoding
        # x: (T)
        x = torch.nn.functional.one_hot(x, input_dim)
        # x: (T, input_dim)
        x = self.logits(x)
        # x: (T, input_dim)
        p = torch.nn.functional.softmax(x, dim=-1)
        # p: (T, input_dim)
        return p

    def generate(self, max_tokens):
        ids = [char_to_i['\n']]
        for _ in range(max_tokens):
            p = self(torch.tensor(ids))[-1, :]
            next_id = torch.multinomial(p, 1).item()
            ids.append(next_id)
        return ''.join(i_to_char[i] for i in ids)


def encode(string: str):
    indices = [char_to_i[c] for c in string]
    return torch.tensor(indices)


def evaluate_split(net: Net, split: str, steps: int=100) -> Net:
    criterion = torch.nn.CrossEntropyLoss()
    data = all_data[split]
    tot = 0
    for _ in range(steps):
        with torch.no_grad():
            position = torch.Random().randint(0, len(all_data) - context_size - 1)
            string = data[position:position + context_size + 1]
            x = string[:, :-1]
            y = string[:, 1:]
            y_hat = net(x)
            loss = criterion(y_hat, y)
            tot += loss.item()
    print(tot / steps)
    return net


def train(net, steps=10000) -> Net:
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(steps):
        position = torch.Random().randint(0, len(all_data) - context_size - 1)
        string = all_data['train'][position:position + context_size + 1]
        x = string[:, :-1]
        y = string[:, 1:]
        y_hat = net(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % print_loss_every_n_steps == 0:
            net = evaluate_split(net, 'train')
            net = evaluate_split(net, 'test')
    return net


def main():
    net = train(Net())
    print(net.generate(1000))
