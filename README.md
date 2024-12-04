# Minority Language AI Interactive Learning Platform – The Case of Siouguluan Amis Language

## Team
- CS 4 [__潘煜智__](https://github.com/YCNeo718)
- CS 3 [__黃蓉容__](https://github.com/Zhong220)

## Run
```bash
# run without docker
$ pip install gradio
$ python3 app.py

# run with docker
$ docker build --rm -t gradio-app .
$ docker run --name AIP-gradio -p 7860:7860 gradio-app

# clean unused
$ docker container prune -f && docker image prune -f
```

## Training code
- [Colab](https://colab.research.google.com/drive/1vrkRt4QuIxehGeQKRRDML92yEOJFdjR5?usp=sharing)