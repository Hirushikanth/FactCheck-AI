# Extractor Agent

The subgraph runs six stages:

1. Sentence splitting with context windows.
2. Selection of sentences containing specific, verifiable propositions.
3. Disambiguation of resolvable pronouns, references, acronyms, and structural ambiguities.
4. Decomposition into atomic factual claims.
5. Fidelity checks that extracted claims faithfully represent the source sentence.
6. Validation that claims are complete declarative sentences.
