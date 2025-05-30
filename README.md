# PRINT_sequence_model
Extracting semantic insights from PRINT datasets

PRINT_sequence_model/
│
├── data/                        # Data storage
│   ├── SBB_data.docx            # Original document
│   ├── parsed_records.csv       # Parsed data in CSV format
│   ├── parsed_records.parquet   # Parsed data in Parquet format
│   └── embeddings/              # Stored embeddings
│       └── records_with_embeddings.parquet
│
├── models/                      # Cached models
│   └── sentence_transformer/    # Downloaded transformer models
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── data_processing.py       # Functions for data processing
│   ├── embedding_utils.py       # Embedding generation utilities
│   └── search_engine.py         # Search functionality
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── data_exploration.ipynb
│   └── search_analysis.ipynb
│
├── parse_initial_data.py        # Script to parse docx to structured data
├── generate_embeddings.py       # Script to create embeddings
├── search_topics.py             # Script to perform searches
│
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation