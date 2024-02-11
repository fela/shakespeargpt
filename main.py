import torch
import datasets

raw_data = datasets.load_dataset('tiny_shakespeare')
all_data = {
    'train': raw_data['train']['text'][0],
    'test': raw_data['test']['text'][0]
}

chars = ''.join(set(all_data['train']+all_data['test']))
input_dim = len(chars)
char_to_i = {c: i for i, c in enumerate(chars)}
i_to_char = {i: c for i, c in enumerate(chars)}
context_size = 8
print_loss_every_n_steps = 100


class Net(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            # torch has no attribute Random
            position = torch.randint(low=0, high=len(data) - context_size - 1, size=(1,)).item()
            string = data[position:position + context_size + 1]
            inp = encode(string)
            x = inp[:-1]
            y = inp[1:]
            y_hat = net(x)
            loss = criterion(y_hat, y)
            tot += loss.item()
    print(tot / steps)


def train(net, steps=10000) -> Net:
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    for i in range(steps):
        if i % print_loss_every_n_steps == 0:
            for split in ['train', 'test']:
                print(f'Loss on {split} at step {i}: ', end='')
        position = torch.randint(low=0, high=len(all_data['train']) - context_size - 1, size=(1,)).item()
        string = all_data['train'][position:position + context_size + 1]
        inp = encode(string)
        x = inp[:-1]
        y = inp[1:]
        y_hat = net(x)
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net


def main():
    net = train(Net())
    print(net.generate(1000))


if __name__ == '__main__':
    main()
