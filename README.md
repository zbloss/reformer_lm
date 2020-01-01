# Reformer
a Pytorch implementation of the Reformer Network (https://openreview.net/pdf?id=rkgNKkHtvB)

Much of this code base is loosely translated from the jax implementation found here from Google: [https://github.com/google/trax/blob/master/trax/models/research/reformer.py](https://github.com/google/trax/blob/master/trax/models/research/reformer.py)

# How to use
All of the hard work has been taken care of, all you need to do is instantiate the model!

```
from reformer_lm.reformer_lm import ReformerLM
import torch

test = torch.rand((4, 4, 64))
model = ReformerLM(
    vocab_size=300000,
    d_in=test.shape[-2],
    d_out=test.shape[-1],
    n_layers=6,
    n_heads=1,
    attn_k=test.shape[-1],
    attn_v=test.shape[-1],
)

output = model(test)
print(output)

```

This model is still in testing, and will therefore continue to see updates. PRs are welcomed! Feel free to take advantage of the Docker container for development. I have been working in notebooks to test code with the original paper, and then I refactor my code back into the package


[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=7TZ7CL23G7BCQ&currency_code=USD&source=url)
