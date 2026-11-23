# incidental-findings

Reference code for the following works:

Barlow SH, Chicklore S, He Y, Ourselin S, Wagner T, Barnes A and Cook GJR (2025) Open LLM-based actionable incidental finding extraction from [18F]fluorodeoxyglucose PET-CT radiology reports. Front. Digit. Health 7:1702082. doi: 10.3389/fdgth.2025.1702082

Please see https://github.com/stephenhbarlow/inference-time-verification for an updated implementation of verifier training and inference.

train_generator.py - provides code for training a 'Generator' model for both papers.

create_verification_dataset.py - provides code for sampling data in order to train a verifier model.

train_verifier.py - provides code for training a verifier model.

standard_inference.py, verified_inference.py, and majority_vote_inference.py provide code for the different inference strategies.


Due to data privacy concerns the original datasets cannot be stored in this repo. The code is provided to show how the results were acheived.
