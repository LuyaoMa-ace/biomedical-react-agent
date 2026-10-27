# Fixed chain (baseline_vanilla_rag_v3) vs dynamic agent (agent_full_granite_guardlocal)

Questions where EM differs: agent better 16, agent worse 26 (of 500).


## agent BETTER than the fixed chain

### qid 71  (category 5)
**Question:** Which gene encodes Ig-alpha, a component of the pre-B cell receptor complex?

**Gold:** mb-1

**Fixed chain:** pred = `mb-1 gene`  (EM 0), gold paper rank 2

**Agent:** pred = `Ig-alpha`  (EM 1); stop = final; steps = 2; tools = {'retrieve_from_index': 1}; gold paper seen = False

- step 1 [action] `retrieve_from_index(Ig-alpha gene)`  — thought: I should search the local PubMed corpus for information about Ig-alpha and its encoding gene.  → obs: '[1] Long-range genomic interactions epigenetically regulate the expression of a cytokine r'
- step 2 [final]  — thought: I now know the final answer

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 333  (category 1)
**Question:** Which gene is responsible for producing the catalytic region of aspartyl-tRNA synthetase in rat hepatic tissue?

**Gold:** DRS1

**Fixed chain:** pred = `DRS1 gene`  (EM 0), gold paper rank 1

**Agent:** pred = `DRS1`  (EM 1); stop = final; steps = 4; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(Which gene is responsible for producing the catalytic region of aspartyl-tRNA sy)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: gene responsible for catalytic region of aspartyl-tRNA synthetase in rat hep'
- step 2 [action] `generate_reasoning_query(Which gene is responsible for producing the catalytic region of aspartyl-tRNA sy)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question asks about a specifi'
- step 3 [action] `retrieve_from_index(gene responsible for catalytic region of aspartyl-tRNA synthetase in rat hepatic)`  — thought: I should now search the local corpus using the generated reasoning query.  → obs: '[1] Molecular cloning and primary structure of cDNA encoding the catalytic domain of rat l'
- step 4 [final]  — thought: The first abstract mentions the gene DRS1 encoding the catalytic domain of rat liver aspartyl-tRNA synthetase.

*Auto notes:* none

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 137  (category 1)
**Question:** Which transgenic mouse line possesses a heat-sensitive alteration in the SV40 large T antigen gene, regulated by the H-2K(b) promoter that responds to interferon gamma?

**Gold:** H-2K(b)-tsA58 transgenic mouse (immortomouse)

**Fixed chain:** pred = `H-2K(b)-tsA58 transgenic mice`  (EM 0), gold paper rank 1

**Agent:** pred = `Immortomice`  (EM 1); stop = final; steps = 3; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(heat-sensitive alteration SV40 large T antigen gene H-2K(b) promoter interferon )`  — thought: I should reformulate the question into a keyword-style search query.  → obs: 'Search query: heat-sensitive alteration SV40 large T antigen gene H-2K(b) promoter interfe'
- step 2 [action] `generate_reasoning_query(heat-sensitive alteration SV40 large T antigen gene H-2K(b) promoter interferon )`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question involves the interac'
- step 3 [final]  — thought: I now know the final answer
- step 4 [guardrail] `retrieve_from_index(heat-sensitive alteration SV40 large T antigen gene H-2K(b) promoter interferon )`  → obs: '[1] Characterization of fetal and postnatal enteric neuronal cell lines with improvement i'

*Auto notes:* none

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?


## agent WORSE than the fixed chain

### qid 189  (category 3)
**Question:** Which pair of substances are noted for their interaction within the CNS, influencing the regulation of neurotransmitter activity?

**Gold:** Angiotensin and acetylcholine

**Fixed chain:** pred = `Angiotensin and acetylcholine`  (EM 1), gold paper rank 1

**Agent:** pred = `P2X5 receptors`  (EM 0); stop = final; steps = 3; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'retrieve_from_index': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(pair of substances noted for their interaction within the CNS, influencing the r)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: CNS neurotransmitter regulation interaction substances pair'
- step 2 [action] `generate_reasoning_query(pair of substances noted for their interaction within the CNS, influencing the r)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question is about the interac'
- step 3 [final]  — thought: I now know the final answer
- step 4 [guardrail] `retrieve_from_index(CNS neurotransmitter regulation interaction substances pair)`  → obs: '[1] Effect of the cerebral tryptaminergic system on the turnover of dopamine in the striat'

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 319  (category 5)
**Question:** Which bacterial pathogen is known for multidrug resistance and causes infections in immunocompromised patients?

**Gold:** Acinetobacter baumannii

**Fixed chain:** pred = `Acinetobacter baumannii`  (EM 1), gold paper rank 46

**Agent:** pred = `Pseudomonas aeruginosa`  (EM 0); stop = final; steps = 4; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'fetch_pubmed_live': 1, 'retrieve_from_index': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(bacterial pathogen known for multidrug resistance and causes infections in immun)`  — thought: I should reformulate the question into a keyword-style search query.  → obs: 'Search query: bacterial pathogen multidrug resistance immunocompromised patients infection'
- step 2 [action] `generate_reasoning_query(bacterial pathogen known for multidrug resistance and causes infections in immun)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question asks about a specifi'
- step 3 [action] `fetch_pubmed_live(Pseudomonas aeruginosa multidrug resistance immunocompromised)`  — thought: I should fetch more specific keywords from PubMed.  → obs: '[1] Antibiotic resistance in Pseudomonas aeruginosa: mechanisms and alternative therapeuti'
- step 4 [final]  — thought: I now know the final answer
- step 5 [guardrail] `retrieve_from_index(bacterial pathogen multidrug resistance immunocompromised patients infections)`  → obs: '[1] The Role and Mechanism of Thiol-Dependent Antioxidant System in Bacterial Drug Suscept'

*Auto notes:* gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 39  (category 2)
**Question:** What is the recommended initial treatment for septic arthritis following anterior cruciate ligament reconstruction?

**Gold:** Culture-specific intravenous antibiotics and surgical joint irrigation with graft retention

**Fixed chain:** pred = `Culture-specific IV antibiotics and surgical joint irrigation with graft retention`  (EM 1), gold paper rank 1

**Agent:** pred = `Culture-specific intravenous (IV) antibiotics`  (EM 0); stop = final; steps = 3; tools = {'query_reformulator': 1, 'generate_reasoning_query': 1, 'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `query_reformulator(recommended initial treatment for septic arthritis following anterior cruciate l)`  — thought: I should first reformulate the question into a keyword-style search query.  → obs: 'Search query: recommended initial treatment for septic arthritis following anterior crucia'
- step 2 [action] `generate_reasoning_query(recommended initial treatment for septic arthritis following anterior cruciate l)`  — thought: I should generate a reasoning query to guide the search.  → obs: 'Reasoning (use it as the query for retrieve_from_index): The question asks for the standar'
- step 3 [final]  — thought: I now know the final answer
- step 4 [guardrail] `retrieve_from_index(recommended initial treatment for septic arthritis following anterior cruciate l)`  → obs: '[1] Septic arthritis of the knee following anterior cruciate ligament reconstruction: resu'

*Auto notes:* none

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?
