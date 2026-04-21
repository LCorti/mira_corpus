# Directory structure

```
raw_data
+--Development-Set-Apr8             // 10 reference papers X citing papers
|   +--[ACL Anthology Paper ID]
|   |   +--annotation
|   |   +--Citance_XML              // Citing papers
|   |   +--Reference_XML            // Reference paper
|   |   +--summary                  // Summaries (abstract, citances, human)
|   +--[Another paper]
|
+--Test-Set-2016
|   +--[ACL Anthology Paper ID]
|   |   +--annotation
|   |   +--[ACL Anthology Paper ID] // Citance_XML + Reference_XML ??
|   |   +--Citance_XML              // Citing papers
|   |   +--Reference_XML            // Reference paper
|   |   +--summary                  // Summaries (abstract, citances, human)
|   +--[Another paper]
|
+--Test-Set-2017
|   +--[Same as Test-Set-2016]
|
+--Test-Set-2018
|   +--[ACL Anthology Paper ID]
|   |   +--annotation
|   |   +--[ACL Anthology Paper ID] // PDF of reference paper
|   |   +--Citance_XML              // Citing papers
|   |   +--Reference_XML            // Reference paper
|   |   +--summary                  // Copied from Test-Set-2019-Gold
|   +--[Another paper]
|
+--Training-Set-2016
|   +--[Same as Test-Set-2016 but without single folder with XMLs]
|
+--Training-Set-2017
|   +--[Same as Test-Set-2016]
|
+--Training-Set-2018
|   +--[Same as Test-Set-2016]
|
+--Training-Set-2019
|   +--From-ScisummNet-2019
|   |   +--[ACL Anthology Paper ID]
|   |   |   +--Reference_XML    // The paper
|   |   |   +--summary          // Abstract + summary
|   |
|   +--From-Training-Set-2018   // Same training data
|   |   |
```

