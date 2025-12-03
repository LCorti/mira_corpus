"""
Relevance indices for document retrieval evaluation.

Each list contains the paper indices (0-based positions in selected_papers.jsonl)
that can answer the question at that position.

Generated automatically from LLM answerability judgments.
"""

relevance_indices = [
    [0,1,2,3,4,6,8,9],              # 1
    [0,4,6],                        # 2
    [0],                            # 3
    [9],                            # 4
    [1],                            # 5
    [0,1,6],                        # 6
    [0,1,2,4,6],                    # 7
    [2],                            # 8
    [0,2,4],                        # 9
    [3],                            # 10
    [3],                            # 11
    [3],                            # 12
    [4,6],                          # 13
    [4],                            # 14
    [4],                            # 15
    [5],                            # 16
    [5],                            # 17
    [5],                            # 18
    [0,1,2,4,6],                    # 19
    [2,6],                          # 20
    [6],                            # 21
    [0,4,5,6,7,8],                  # 22
    [7],                            # 23
    [7],                            # 24
    [8],                            # 25
    [8],                            # 26
    [8],                            # 27
    [9],                            # 28
    [9],                            # 29
    [9],                            # 30
]
