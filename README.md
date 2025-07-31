Overview
==========
This site contains the course materials for the [**ESSLLI 2025**](https://2025.esslli.eu/) course of **Language Model Programming**.

**overview** When developing complex AI systems, which nowadays couple many individual components and tools (e.g., code interpreters, search technology) with **large language models** (LLMs) at the core, a natural question arises: *how can users and developers of these systems compose model components into a coherent system to best achieve their goals?* Such is the problem of **language model programming**, which concerns how the underlying components of AI systems are assembled, the nature of the interaction between components, and, importantly, the *language that users use to specify the design and implementation of these systems*. In this course, we explore this emerging literature on model language programming and attempts to relate LLM development to conventional programming. We specifically look at the fundamentals of how to build high-level modeling languages for LLMs, the different *paradigms* of model programming that exist (e.g., [functional](https://arxiv.org/pdf/2106.06981) vs  [imperative](https://arxiv.org/abs/2212.06094) vs. [declarative](https://dl.acm.org/doi/abs/10.1145/3591280) vs. [probabilistic](https://arxiv.org/pdf/2207.10342)) and the problems in NLP that they address and aim to solve (e.g., theoretical understanding of transformers, model fine-tuning, preference alignment, advanced prompting and constrained decoding). 

(See [here](https://github.com/yakazimir/esslli_2024_llm_programming) for last year's version of this course. Some content is also taken from our ESSLLI 2023 course on [neuro-symbolic modeling](https://github.com/yakazimir/esslli_neural_symbolic); see there for additional pointers). 

Lecturers 
==========

[**Kyle Richardson**](https://www.krichardson.me/) (Allen Institute for AI) 

[**Gijs Wijnholds**](https://gijswijnholds.github.io/) (Leiden Institute of Advanced Computer Science)

Slides 
==========

[**lecture 1**](https://github.com/yakazimir/LMProgramming/blob/main/slides/lecture1.pdf): course overview, **language modeling basics**, [**transformers**](https://arxiv.org/abs/1706.03762),  [**RASP**](https://arxiv.org/pdf/2106.06981).

[**lecture 2**](https://github.com/yakazimir/LMProgramming/blob/main/slides/lecture2.pdf): declarative approaches to **model training and fine-tuning**, the [**semantic loss**](https://arxiv.org/pdf/1711.11157) and [**weighted model counting**](https://www.sciencedirect.com/science/article/pii/S0004370207001889),  [**other**](https://arxiv.org/abs/1909.00126) approaches.

[**lecture 3**](https://github.com/yakazimir/LMProgramming/blob/main/slides/lecture3.pdf): high-level programming techniques for [**direct preference alignment**](https://arxiv.org/abs/2305.18290) and [**LLM alignment**](https://www.jair.org/index.php/jair/article/view/17541), [**formal characterizations**](https://arxiv.org/abs/2412.17696) of *known* loss functions.  

[**lecture 4**](https://github.com/yakazimir/LMProgramming/blob/main/slides/lecture4.pdf): [**declarative and probabilistic approaches**](https://www.khoury.northeastern.edu/home/lieber/courses/csg260/f06/materials/papers/bayes/AAAI02-102.pdf) to test-time inference, [**LLM self-correction**](https://arxiv.org/abs/2211.11875), [**consistency**](https://arxiv.org/pdf/2409.13724), distilling LLMs to tractable models, [**logic programming**](https://arxiv.org/abs/1805.10872).

**lecture 5**: Advanced prompting, [**chain-of-thought**](https://proceedings.neurips.cc/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html), [**imperative model programming**](https://arxiv.org/pdf/2212.06094), ([discrete](https://arxiv.org/abs/1904.02079)) [**probabilistic programming**](https://discovery.ucl.ac.uk/id/eprint/10089698/1/main.pdf).


**background** [logic notes](https://github.com/yakazimir/LMProgramming/blob/main/slides/logic_primer.pdf), [extended notes on transformers](https://www.krichardson.me/files/lms.pdf)

**extra lectures** [Prompting as programming](https://github.com/yakazimir/LMProgramming/blob/main/slides/prompting_programming.pdf), [Grammar-constrained decoding](https://github.com/yakazimir/LMProgramming/blob/main/slides/grammar_decoding.pdf)

Helpful Resources 
==========

Below are some pointers to code resources:
- **languages** [[scallop]](https://github.com/scallop-lang/scallop), [[problog]](https://github.com/ML-KULeuven/problog), [[pyDatalog]](https://github.com/pcarbonn/pyDatalog), [[lmql]](https://github.com/eth-sri/lmql),[[rasp]](https://github.com/tech-srl/RASP), [[NumPy Rasp]](https://github.com/apple/ml-np-rasp), [[deepproblog]](https://github.com/ML-KULeuven/deepproblog) 
- **automated reasoning tools** [[Z3 solver]](https://github.com/Z3Prover/z3), [[python-sat]](https://pysathq.github.io/), [[pysdd]](https://github.com/wannesm/PySDD) 
- **NLP and general ML** [[transformers]](https://github.com/huggingface/transformers), [[PyTorch]](https://pytorch.org/), [[pylon-lib]](https://github.com/pylon-lib/pylon), [[hf datasets]](https://huggingface.co/docs/datasets/index)
- **other useful utilities** [[sympy]](https://www.sympy.org/en/index.html)

**Useful tutorials**: [**Transformers from scratch**](https://peterbloem.nl/blog/transformers) (*some examples/ideas used in lecture 1*), [**Lectures on Probabilistic Programming**](https://www.khoury.northeastern.edu/home/sholtzen/oplss24-ppl/), [**Tractable Probabilistic Models**](https://web.cs.ucla.edu/~guyvdb/slides/TPMTutorialUAI19.pdf)