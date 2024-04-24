# Analysis

## Interesting Cases
1. "response": "The Speaker of the House of Commons presides over the joint sessions of parliament."

"triplet": [
    "Speaker of the House of Commons",
    "presides over",
    "joint sessions of parliament"
]

"human_label": "Contradiction"
"nli_label": "Contradiction"

Explanation: When the question is generic (in this case, the country or the system is not specified) the model might not reply accurately, and since we have our reference, which is specific to the wikipedia entry, this is an issue. 

2. "context": "John Brown 's raid on Harper's Ferry ( also known as John Brown 's raid or The raid on Harper 's Ferry ) was an effort by armed abolitionist John Brown to initiate an armed slave revolt in 1859 by taking over a United States arsenal at Harpers Ferry , Virginia . Brown 's party of 22 was defeated by a company of U.S. Marines , led by First Lieutenant Israel Greene . Colonel Robert E. Lee was in overall command of the operation to retake the arsenal . John Brown had originally asked Harriet Tubman and Frederick Douglass , both of whom he had met in his transformative years as an abolitionist in Springfield , Massachusetts , to join him in his raid , but Tubman was prevented by illness and Douglass declined , as he believed Brown 's plan would fail."

"triplet": [
    "Colonel Robert E. Lee",
    "led",
    "soldiers"
],

"human_label": "Contradiction"
"nli_label": "Entailment"

Explanation: In this case, the NLI causes some issues due to the fact that a simple NLI task can't be used as a placeholder method for an actual method, which is why we used an LLM based output. 

**"llm_label": "Contradiction"**

## Drawbacks and Insights:
1. Depening on an LLM, means we can never be sure of the kind of outputs it will produce, for example it repeats the same Q/A pair multiple times:

python
```
    {
        "triplet": [
            "soldiers",
            "ended",
            "raid on the harper's ferry arsenal"
        ],
        "human_label": "Entailment"
    },
    {
        "triplet": [
            "raid on the harper's ferry arsenal",
            "location",
            "harper's ferry"
        ],
        "human_label": "Entailment"
    },
    {
        "triplet": [
            "Colonel Robert E. Lee",
            "led",
            "soldiers"
        ],
        "human_label": "Contradiction"
    },
    {
        "triplet": [
            "soldiers",
            "ended",
            "raid on the harper's ferry arsenal"
        ],
        "human_label": "Entailment"
    }
```

2. The LLM might not generate the all the knowledge graphs that we need:
Answer: 
"response": "Charkie is a fictional character from the children's book series Curious George. He is a yellow Labrador Retriever.",

python
```
    [
    "Charkie",
    "is a",
    "fictional character"
    ],
    [
    "Charkie",
    "from",
    "children's book series Curious George"
    ],
    [
    "Charkie",
    "is a",
    "yellow Labrador Retriever"
    ]
```

Context: "Charkie : A female black cocker spaniel owned by Steve and Betsy . Charkie is very hyperactive , loves doing backflips , and frequently runs away from her caretakers . She is very skilled in opening and unlocking things ."

Looking at it deeply, we realise that Charkie, is mentioned to be a female in the context, but the LLM generates the input containing a "He", and usually we miss this sort of granularity in the LLM generated KG's, and we can't be sure, that we generate all the KG pairs in the given 
