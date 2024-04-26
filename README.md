# GPT1: from OpenAI paper 'Improving Language Understanding by Generative Pre-Training'

## Paper analysis
It demonstrates that large gains on diverse tasks such as textual entailment, question answering, semantic similarity assessment, 
and document classification can be realized by generative pre-training of a language model on a diverse corpus of unlabeled text, 
followed by discriminative fine-tuning on each specific task. In contrast to previous approaches, it makes use of task-aware input
transformations during fine-tuning to achieve effective transfer while requiring minimal changes to the model architecture. 

The goal is to learn a universal representation that transfers with little adaptation to a wide range of tasks. 
The setup does not require these target tasks to be in the same domain as the unlabeled
corpus. We employ a two-stage training procedure. First, we use a language modeling objective on
the unlabeled data to learn the initial parameters of a neural network model. Subsequently, we adapt
these parameters to a target task using the corresponding supervised objective. 

For our model architecture, we use the Transformer, During transfer, we utilize task-specific input adaptations derived 
from traversal-style approaches which process structured text input as a single contiguous sequence of tokens.


## FRAMEWORK

Our training procedure consists of two stages. 
### Generative Pretraining
1. The first stage is learning a high-capacity language model on a large corpus of text.
    Given an unsupervised corpus of tokens U = {u1, . . . , un}, we use a standard language modeling (a language model is
    a probability distribution describing the likelihood of any string) objective. These parameters are trained using 
    stochastic gradient descent. We use a multi-layer Transformer decoder. This model applies a multi-headed self-attention 
    operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over 
    target tokens. 
    
    We use the BooksCorpus dataset [71] for training the language model. It contains over 7,000 unique unpublished 
    books from a variety of genres including Adventure, Fantasy, and Romance. Crucially, it contains long stretches of contiguous 
    text, which allows the generative model to learn to condition on long-range information. Our language model achieves a 
    very low token level perplexity of 18.4 on this corpus. 
    Perplexity formula: PP(S) = 1/(P(S))^1/n where n is the number of tokens in S.

    Our model largely follows the original transformer work. We trained a
    12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12
    attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.
    We used the Adam optimization scheme [27] with a max learning rate of 2.5e-4. The learning rate
    was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule.
    We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
    Since layernorm [2] is used extensively throughout the model, a simple weight initialization of
    N(0,0.02) was sufficient. We used a bytepair encoding (BPE) vocabulary with 40,000 merges [53]
    and residual, embedding, and attention dropouts with a rate of 0.1 for regularization. We also
    employed a modified version of L2 regularization proposed in [37], with w = 0.01 on all non bias or
    gain weights. For the activation function, we used the Gaussian Error Linear Unit (GELU) [18]. We
    used learned position embeddings instead of the sinusoidal version proposed in the original work.
    We use the ftfy library2 to clean the raw text in BooksCorpus, standardize some punctuation and
    whitespace, and use the spaCy tokenizer
    
### Finetuning

2. This is followed by a fine-tuning stage, where we adapt the model to a discriminative task with labeled data.
    After training the model with the language modeling objective, we adapt the parameters to the supervised target task. 
    We assume a labeled dataset C, where each instance consists of a sequence of input tokens,x1,...,xm, along with a label y.
    The inputs are passed through our pre-trained model to obtain the final transformer block’s activation hml , which is then fed 
    into an added linear output layer with parameters Wy to predict y: P(y|x1,...,xm) = softmax(hml Wy). 
    We additionally found that including language modeling as an auxiliary objective to the fine-tuning helped learning by 
    (a) improving generalization of the supervised model, and 
    (b) accelerating convergence.
    Specifically, we optimize the following objective (with weight λ):
    L3(C) = L2(C) + λ ∗L1(C) (5). 
    Overall, the only extra parameters we require during fine-tuning are Wy, and embeddings for delimiter tokens.
    ----Task specific input Transformations----------
    - Classification: For some tasks, like text classification, we can directly fine-tune our model as described above.
    - Textual entailment: For entailment tasks, we concatenate the premise p and hypothesis h token sequences, 
        with a delimiter token ($) in between.
    - Similarity: For similarity tasks, there is no inherent ordering of the two sentences being compared. To reflect this, we modify 
        the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each independently 
        to produce two sequence representations which are added element-wise before being fed into the linear output layer.
    - Question Answering and Commonsense Reasoning: For these tasks, we are given a context document z, a question q, 
        and a set of possible answers {ak}. We concatenate the document context and question with each possible answer, 
        adding a delimiter token in between to get [z; q; $; ak]. Each of these sequences are processed independently with 
        our model and then normalized via a softmax layer to produce an output distribution over possible answers.  
    Unless specified, we reuse the hyperparameter settings from unsupervised pre-training. 
    We add dropout to the classifier with a rate of 0.1. For most tasks, we use a learning rate of 6.25e-5 and a 
    batchsize of 32. Our model finetunes quickly and 3 epochs of training was sufficientor most cases. We use a linear 
    learning rate decay schedule with warmup over 0.2% of training. λ was set to 0.5.
     ---

