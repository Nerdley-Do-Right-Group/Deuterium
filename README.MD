# Codename Deuterium
**Track:** Finance

**Team:** Luke, Sawyer, and Joel
## The problem
Many people have to unfortunately deal with spam messages. The most common target for such messages are the elderly, however, anyone found in a data breach has to deal with these messages. Sometimes, especially for the elderly, they can be very tricky. Thus proposes the problem: how can we prevent such messages from annoying and even initiating a transaction where the victim gets stolen from?
## What we built
What goes in, what comes out, and how someone would use it.

Text is fed into this program via the `predict` command which will return its verdict and confidence. To train the model run the `train` command and it will produce a model. To setup your environment run one of our setup scripts. The program uses machine learning from the `sklearn` library to produce a model and predict based on that model.

## What we added
Starting from SMSSpam filter, we added about 4,000 lines and exaples for training the ai of various forms, all from real life spam calls, as well as examples of normal (ham) messages.
## How it works
Data: which dataset, how many examples, how we cleaned it.
Model: what we used and why.
## Results
Our f1 baseline was 86%, and our final is 98.03%.
It may fail if it gets a message its database does not cover. 
## What we'd do next
add a gui for integrations in chats, testing, and for users to edit settings.
## Sources
Datasets, pretrained models, tutorials and AI tools used, with links.

> Almeida, Tiago and Jos Hidalgo. "SMS Spam Collection." UCI Machine Learning Repository, 2011, https://doi.org/10.24432/C5CC84.

> Google Gemini Models (for the setup scripts and comments)

> Qwen Local Models (for generating select additional entries to the dataset)

> scikit-learn python package

> joblib python package

> pytest python package