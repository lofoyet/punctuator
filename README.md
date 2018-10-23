# Punctuator by @ottokart

Bomoda borrows a model to add missing punctuations back to documents, especially radio transcripts.
Check out https://github.com/ottokart/punctuator2

## How to use

```python
import sys
sys.path.append("bomoda2/fireworks")

# define your own tokenize function
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

from lib.punctuator import Punctuator
P = Punctuator(
    tokenize_func=tknzr.tokenize
    )
P.load()
P.punctuate("hi this is the best-looking guy on globe why you laugh get lost")
```
