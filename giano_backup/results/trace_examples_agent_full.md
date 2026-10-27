# Fixed chain (baseline_vanilla_rag_v3) vs dynamic agent (agent_full)

Questions where EM differs: agent better 16, agent worse 56 (of 500).


## agent BETTER than the fixed chain

### qid 145  (category 4)
**Question:** Which classic biochemical technique is commonly employed to locate RNA segments that bind proteins by shielding those areas from enzymatic digestion or chemical alteration?

**Gold:** Footprinting

**Fixed chain:** pred = `Footprinting technique`  (EM 0), gold paper rank 1

**Agent:** pred = `Footprinting`  (EM 1); stop = final; steps = 3; tools = {'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(RNA binding protein localization technique shielding from digestion)`  — thought: I need to find the answer to this question. I will use the local corpus first.  → obs: '[1] Site-Directed Chemical Probing to map transient RNA/protein interactions.\nRNA-protein '
- step 2 [action] `retrieve_from_index(RNA binding protein localization technique shielding from digestion)`  — thought: The abstracts discuss RNA binding proteins, localization techniques, and RNA interactions. The first abstract describes   → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [final]  — thought: The first abstract describes the Site-Directed Chemical Probing (SDCP) technique, which involves shielding RNA binding p

*Auto notes:* repeated an identical action

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 327  (category 4)
**Question:** What laboratory technique is commonly used to determine the serotype of meningococcal strains?

**Gold:** Serological typing

**Fixed chain:** pred = `Serological typing technique`  (EM 0), gold paper rank 1

**Agent:** pred = `Serological typing`  (EM 1); stop = loop; steps = 4; tools = {'retrieve_from_index': 2}; gold paper seen = True

- step 1 [action] `retrieve_from_index(laboratory technique serotype meningococcal strains)`  — thought: I need to find information about laboratory techniques used to determine the serotype of meningococcal strains. I will u  → obs: '[1] Serotypes of group B meningococci.\nStrains of group B meningococci isolated from cases'
- step 2 [action] `retrieve_from_index(serological typing meningococcal strains)`  — thought: The abstracts discuss different laboratory techniques, including serological typing for meningococci, genetic manipulati  → obs: '[1] Serotypes of group B meningococci.\nStrains of group B meningococci isolated from cases'
- step 3 [action] `retrieve_from_index(serological typing meningococcal strains)`  — thought: The abstracts discuss different laboratory techniques and serotype distributions. The first abstract mentions serologica  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [action] `retrieve_from_index(serological typing meningococcal strains)`  — thought: The abstracts discuss different laboratory techniques and serotype distributions. The first abstract mentions serologica  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 5 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 194  (category 2)
**Question:** During use of the WEB 17 device for intracranial aneurysm therapy, which procedural issue was observed in two cases, though it did not lead to any lasting clinical effects?

**Gold:** Thromboembolic complication

**Fixed chain:** pred = `Thromboembolic procedural complications`  (EM 0), gold paper rank 1

**Agent:** pred = `Thromboembolism`  (EM 1); stop = loop; steps = 3; tools = {'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(WEB 17 device intracranial aneurysm therapy procedural issue two cases no lastin)`  — thought: I need to find abstracts about the WEB 17 device and intracranial aneurysm therapy, specifically looking for procedural   → obs: '[1] The New Low-Profile WEB 17 System for Treatment of Intracranial Aneurysms: First Clini'
- step 2 [action] `retrieve_from_index(WEB 17 device intracranial aneurysm therapy procedural issue two cases no lastin)`  — thought: The question asks about a specific procedural issue observed in two cases during WEB 17 therapy for intracranial aneurys  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `retrieve_from_index(WEB 17 device intracranial aneurysm therapy procedural issue two cases no lastin)`  — thought: The question asks about a specific procedural issue observed in two cases during WEB 17 therapy for intracranial aneurys  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?


## agent WORSE than the fixed chain

### qid 254  (category 5)
**Question:** Which frequently occurring noncancerous growth of the uterus can often resemble a uterine sarcoma on radiological examinations?

**Gold:** Uterine leiomyoma

**Fixed chain:** pred = `Uterine leiomyoma`  (EM 1), gold paper rank 1

**Agent:** pred = `Leiomyoma`  (EM 0); stop = loop; steps = 4; tools = {'retrieve_from_index': 2}; gold paper seen = True

- step 1 [action] `retrieve_from_index(noncancerous uterine growth resembles uterine sarcoma)`  — thought: I need to find abstracts about noncancerous uterine growths that can resemble uterine sarcomas. I will use the local cor  → obs: '[1] Uterine sarcoma: can it be differentiated from uterine leiomyoma with Doppler ultrason'
- step 2 [action] `retrieve_from_index(noncancerous uterine growth resembles uterine sarcoma radiological examination)`  — thought: The abstracts discuss uterine growths, sarcomas, and immunohistochemistry, but none explicitly mention a noncancerous gr  → obs: '[1] Uterine sarcoma: can it be differentiated from uterine leiomyoma with Doppler ultrason'
- step 3 [action] `retrieve_from_index(noncancerous uterine growth resembles uterine sarcoma radiological examination)`  — thought: The question asks for a noncancerous growth of the uterus that can resemble a uterine sarcoma on radiological examinatio  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [action] `retrieve_from_index(noncancerous uterine growth resembles uterine sarcoma radiological examination)`  — thought: The question asks for a noncancerous growth of the uterus that can resemble a uterine sarcoma on radiological examinatio  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 5 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 35  (category 5)
**Question:** Which pre-existing medical condition at the time of breast cancer diagnosis is specifically associated with increased mortality from non-cancer causes?

**Gold:** Cardiovascular disease

**Fixed chain:** pred = `Cardiovascular disease`  (EM 1), gold paper rank 1

**Agent:** pred = `Chronic kidney disease`  (EM 0); stop = loop; steps = 3; tools = {'query_reformulator': 1}; gold paper seen = False

- step 1 [action] `query_reformulator(pre-existing medical condition breast cancer diagnosis mortality non-cancer caus)`  — thought: I need to find abstracts that discuss the relationship between pre-existing medical conditions and mortality from non-ca  → obs: 'Search query: pre-existing medical condition breast cancer diagnosis mortality non-cancer '
- step 2 [action] `query_reformulator(pre-existing medical condition breast cancer diagnosis mortality non-cancer caus)`  — thought: I need to find abstracts that discuss the relationship between pre-existing medical conditions and mortality from non-ca  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `query_reformulator(pre-existing medical condition breast cancer diagnosis mortality non-cancer caus)`  — thought: I need to find abstracts that discuss the relationship between pre-existing medical conditions and mortality from non-ca  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it; gold paper never retrieved

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?

### qid 449  (category 5)
**Question:** What arrhythmia, characterized by a heart rate less than 60 beats per minute, is frequently observed in patients with severe COVID-19 pneumonia?

**Gold:** Sinus bradycardia

**Fixed chain:** pred = `Sinus bradycardia`  (EM 1), gold paper rank 1

**Agent:** pred = `Bradycardia`  (EM 0); stop = loop; steps = 3; tools = {'retrieve_from_index': 1}; gold paper seen = True

- step 1 [action] `retrieve_from_index(bradycardia COVID-19 pneumonia arrhythmia)`  — thought: I need to find abstracts about arrhythmias in COVID-19 pneumonia, specifically focusing on bradycardia.  → obs: '[1] Clinical Characteristics and Outcomes of Patients Hospitalized for COVID-19 Pneumonia '
- step 2 [action] `retrieve_from_index(bradycardia COVID-19 pneumonia arrhythmia)`  — thought: The question asks about bradycardia in COVID-19 pneumonia. The first abstract discusses bradycardia in patients with COV  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 3 [action] `retrieve_from_index(bradycardia COVID-19 pneumonia arrhythmia)`  — thought: The question asks about bradycardia in COVID-19 pneumonia. The first abstract discusses bradycardia in patients with COV  → obs: 'You already ran this exact action. Use a different input or tool, or give the Final Answer'
- step 4 [forced_stop] reason=loop

*Auto notes:* repeated an identical action; loop breaker stopped it

*ANNOTATION (fill in by hand):* where did the agent decide differently, and was it better or worse?
