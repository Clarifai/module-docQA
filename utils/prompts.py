NER_PROMPT = """Using the context, do entity recognition of these texts using PER (person), ORG (organization),
LOC (place name or location), TIME (actually date or year), and MISC (formal agreements and projects) and the Sources (the name of the document where the text is extracted from).
The source can be extract at the end of the context after '\nSource: '.
"Make sure the sure is prefixed with 'Source: ' and is on a new line. Do not include any sources that are part of the text."


FORMAT:
Provide them in JSON format with the following 6 keys:
- PER: {list of people}
- ORG: {list of organizations}
- LOC: {list of locations}
- TIME: {list of times}
- MISC: {list of formal agreements and projects}
- SOURCES: {list of sources}


EXAMPLES:
Here are the definitions with a few examples, do not use these examples to answer the question:
PER (person): Refers to individuals, including their names and titles.
Example:
- Barack Obama, former President of the United States
- J.K. Rowling, author of the Harry Potter series
- Elon Musk, CEO of SpaceX and Tesla

ORG (organization): Refers to institutions, companies, government bodies, and other groups.
Example:
- Microsoft Corporation, a multinational technology company
- United Nations, an intergovernmental organization
- International Red Cross, a humanitarian organization

LOC (place name or location): Refers to geographic locations such as countries, cities, and other landmarks.
Example:
- London, capital of England
- Eiffel Tower, a landmark in Paris, France
- Great Barrier Reef, a coral reef system in Australia

TIME (date or year): Refers to dates, years, and other time-related expressions.
Example:
- January 1st, 2023, the start of a new year
- 1995, the year Toy Story was released

MISC (formal agreements and projects): Refers to miscellaneous named entities that don't fit into the other categories, including formal agreements, projects, and other concepts.
Example:
- Kyoto Protocol, an international agreement to address climate change
- Apollo program, a series of manned spaceflight missions undertaken by NASA
Obamacare, a healthcare reform law in the United States.

Sources (list of sources of the text).
Example:
- Tom Clancy's Jack Ryan
- The New York Times
- Harry Potter and the Sorcerer's Stone
----------------

Before you generate the output, make sure that the named entities are correct and part of the context. 
If the named entities are not part of the context, do not include them in the output.
"""

NER_REFINE_TEMPLATE = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer, including sources: {existing_answer}\n"
    "Do not remove any entities or sources from the existing answer."
    "We have the opportunity to update the list of named entities and sources"
    "(only if needed) with some more context below.\n"
    "Make sure any entities extracted are part of the context below. If not, do not add them to the list."
    "If you see any entities that are not extracted, add them."
    "Use only the context below delimited by triple backticks.\n"
    "------------\n"
    "```{context_str}```\n"
    "------------\n"
    "Given the new context, update the original answer to extract additional entities and sources."
    "Create a more accurate list of named entities and sources."
    "If you do update it, please update the sources as well while keeping the existing sources."
    "The new source can be extracted from the end of the context after 'Source: '"
    "If the context isn't useful, return the existing answer unchanged."
)

NER_QUESTION_TEMPLATE = (
    "Context information is below delimited by triple backticks. \n"
    "---------------------\n"
    "```{context_str}```"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)
