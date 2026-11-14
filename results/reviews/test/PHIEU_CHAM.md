# Phiếu chấm thủ công — test

Nhãn bên dưới do **AI soạn gợi ý** từ trace. Người chấm đọc từng câu: câu trả lời, đoạn được trích dẫn, đáp án chuẩn.
Không đồng ý thì sửa giá trị tương ứng trong `results/reviews/test/<cấu hình>.json`, rồi chạy `review confirm` và `review summarize`.

Thang: answer_grade 0 sai/thiếu · 1 đúng một phần · 2 đúng và đủ (từ chối câu có đáp án = 0). context_grade 0 không có evidence · 1 thiếu điều kiện/nguồn · 2 đủ.

## ag_budget_cite

### t01 — During data chunking, what metadata is added to each chunk so that the chunks can be reassembled into the original dataset?

- **Đáp án chuẩn:** The chunk number and the total number of chunks.
- **Agent (answered):** During data chunking, metadata such as the chunk number and the total number of chunks is added to each chunk. This metadata allows the chunks to be reassembled into the original dataset after being transmitted or stored separately [B:pds-ch02:p100:1].
- **Trích dẫn `B:pds-ch02:p100:1` (trang 100):** These chunks are typically organized sequentially, with each chunk containing a set of data from the larger dataset. The process of data chunking also involves adding metadata (data that provides information about other data), such as the chunk number and the total number of chunks, to each chunk. This metadata allows the chunks to be reassembled into the original dataset after being transmitted or stored separately. Data chunking has several advantages, including the following: 1. Increased speed. By dividing a large dataset into smaller chunks, data processing and transmission can be performed more quickly, reducing the overall processing time. 2. Better utilization of resources. Data chunking enables data to be distributed and processed across multiple machines, making better use of available computing resources. 3. Increased fault tolerance. In case of data corruption or loss, data c
- **Context cuối:** B:pds-ch02:p100:1, B:pds-ch02:p100:0, B:pds-ch02:p100:2
- **Nhãn gợi ý:** answer_grade=2, citation_support=[True], abstention_correct=None, context_grade=2
- **Lý do:** Nêu đúng 'chunk number và total number of chunks'; chunk p100:1 chứa nguyên văn câu này.

### t02 — Which external Python library is mainly used for scraping and parsing HTML and XML code?

- **Đáp án chuẩn:** Beautiful Soup.
- **Agent (answered):** The external Python library mainly used for scraping and parsing HTML and XML code is Beautiful Soup [B:pds-ch02:p80:0].
- **Trích dẫn `B:pds-ch02:p80:0` (trang 80):** In Python, there are several libraries and methods that can be used for parsing and extracting data from text. These include the following: 1. Regular expressions (regex or RE) (https://openstax.org/r/docpython). This is a built-in library in Python that allows for pattern matching and extraction of data from strings. It uses a specific syntax to define patterns and rules for data extraction. 2. Beautiful Soup (https://openstax.org/r/pypi). This is an external library that is mostly used for scraping and parsing HTML and XML code. It can be utilized to extract specific data from web pages or documents. 3. Natural Language Toolkit (NLTK) (https://openstax.org/r/nltk). This is a powerful library for natural language processing in Python. It provides various tools for tokenizing, parsing, and extracting data from text data.
- **Context cuối:** B:pds-ch02:p78:1, B:pds-ch02:p77:0, B:pds-ch02:p80:0
- **Nhãn gợi ý:** answer_grade=2, citation_support=[True], abstention_correct=None, context_grade=2
- **Lý do:** Beautiful Soup đúng; chunk p80:0 ghi rõ 'external library ... scraping and parsing HTML and XML'.

### t03 — Which normal forms are most commonly used in database design?

- **Đáp án chuẩn:** 1NF, 2NF, 3NF (First, Second and Third Normal Form) and BCNF (Boyce-Codd Normal Form).
- **Agent (answered):** The most commonly used normal forms in database design are 1NF (First Normal Form), 2NF (Second Normal Form), 3NF (Third Normal Form), and BCNF (Boyce-Codd Normal Form). These normal forms serve as guidelines to ensure a database is well-structured, organized, and free from redundancy and inconsistency. Normalization involves applying these rules to organize and clean the data effectively [B:pds-ch02:p90:2].
- **Trích dẫn `B:pds-ch02:p90:2` (trang 90):** This may include setting conventions for naming, data types, and formatting A normal form (NF) is a guideline or set of rules used in database design to ensure that a database is well-structured, organized, and free from certain types of data irregularities, such as redundancy and inconsistency. The most commonly used normal forms are 1NF, 2NF, 3NF (First, Second, and Third Normal Form), and BCNF (Boyce-Codd Normal Form). Normalization is the process of applying these rules to a database. The data must be organized and cleaned, which involves removing duplicates and erroneous data, filling in missing values, and logically arranging the data. To uphold data standardization, regular quality control measures should be implemented, including periodic data audits to ascertain the accuracy and consistency of the data. It is also important to document the standardization process, including the 
- **Context cuối:** B:pds-ch02:p90:2, B:pds-ch02:p90:0, B:pds-ch02:p91:1
- **Nhãn gợi ý:** answer_grade=2, citation_support=[True], abstention_correct=None, context_grade=2
- **Lý do:** Đủ 1NF, 2NF, 3NF, BCNF; câu thêm về normalization cũng có trong chunk p90:2.

### t04 — In the United States, how long does copyright protect original creative works?

- **Đáp án chuẩn:** For the life of the creator plus 70 years.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch08:p388:4, B:pds-ch08:p388:3, B:pds-ch08:p389:1, B:pds-ch08:p394:1
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Agent tự viết lại query ('copyright duration...') và lấy p394:1 thay vì p394:0 chứa đáp án; context không có evidence.

### t05 — Which platform is mentioned for sharing data with the broader public in a research context when there are no privacy implications?

- **Đáp án chuẩn:** Kaggle, which is designed for open data sharing in a research context.
- **Agent (answered):** The platform mentioned for sharing data with the broader public in a research context when there are no privacy implications is Kaggle [B:pds-ch08:p397:1].
- **Trích dẫn `B:pds-ch08:p397:1` (trang 397):** As introduced in Python Basics for Data Science and discussed through this book, Google Colab (https://openstax.org/r/colab), which is connected to Google Drive, allows users to collaborate remotely on coding tasks. Files may be hosted on Google Drive or Microsoft OneDrive, allowing lead data scientists to set different levels of permissions relevant to each team member or outside entity. Those directly involved in the project, including team members, customers, researchers, and the project's eventual audience, should be granted access to the data in accordance with the intended purpose. If appropriate, and with no privacy-related implications, the data may be disseminated to the broader public through platforms like Kaggle (https://openstax.org/r/kaggle), which is designed for open data sharing in a research context.
- **Context cuối:** B:pds-ch08:p397:0, B:pds-ch08:p396:3, B:pds-ch08:p397:1
- **Nhãn gợi ý:** answer_grade=2, citation_support=[True], abstention_correct=None, context_grade=2
- **Lý do:** Kaggle đúng; chunk p397:1 chứa 'platforms like Kaggle ... open data sharing in a research context' kèm điều kiện 'no privacy-related implications'.

### t06 — In the plant growth experiment on sunlight, how many hours of sunlight per day did each of the two groups receive?

- **Đáp án chuẩn:** One group received eight hours of sunlight per day and the other only four hours.
- **Agent (answered):** In the plant growth experiment on sunlight, one group of plants received eight hours of sunlight per day, while the other group received only four hours of sunlight per day. This setup was used to investigate whether increased exposure to sunlight enhances plant growth [B:pds-ch02:p71:1][B:pds-ch02:p71:0].
- **Trích dẫn `B:pds-ch02:p71:1` (trang 71):** The main research objective was to determine the growth rate of plants exposed to eight hours of sunlight compared to those with only four hours. A total of 20 identical potted plants were used, with one group allocated to the "sunlight" condition and the other to the "limited sunlight" condition. Both groups were maintained under identical environmental conditions, including temperature, humidity, and soil moisture. Adequate watering was provided to ensure equal hydration of all plants. The measurements of plant height were obtained and accurately recorded every week. This approach allowed for the collection of precise and reliable data on the impact of sunlight on plant growth, which can serve as a valuable resource for further research and understanding of this relationship.
- **Trích dẫn `B:pds-ch02:p71:0` (trang 71):** Collecting Data Through Experiments Collecting data through scientific experiments requires a well-designed experimental scheme, describing the research objectives, variables, and procedures. The establishment of a control specimen is crucial, and data is obtained through systematic properties, measurements, or characteristics. It is crucial to follow ethical guidelines for the proper documentation and ethical utilization of the collected data (see Ethics in Data Collection). Consider this example: Scientist Sally aimed to investigate the impact of sunlight on plant growth. The research inquiry was to determine whether increased exposure to sunlight enhances the growth of plants. Sally experimented with two groups of plants wherein one group received eight hours of sunlight per day, while the other only received four hours. The height of each plant was measured and documented every week 
- **Context cuối:** B:pds-ch02:p71:1, B:pds-ch02:p71:0, B:pds-ch02:p70:0
- **Nhãn gợi ý:** answer_grade=2, citation_support=[True, True], abstention_correct=None, context_grade=2
- **Lý do:** 8 giờ và 4 giờ đúng; p71:0 có câu gold, p71:1 nhắc lại 'eight hours ... compared to those with only four hours' nên cả 2 citation đều hỗ trợ.

### u03 — What sample size is needed for a survey to reach a 95% confidence level with a 5% margin of error?

- **Đáp án chuẩn:** (câu thiếu evidence) Mục 2.2 nói về sampling và sample size nói chung, không có công thức/số mẫu cho mức tin cậy 95%.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch02:p75:1, B:pds-ch02:p73:4, B:pds-ch02:p76:0, B:pds-ch02:p75:0
- **Nhãn gợi ý:** answer_grade=None, citation_support=[], abstention_correct=True, context_grade=None
- **Lý do:** Từ chối, không đưa ra con số nào; corpus không có công thức/số mẫu cho 95% và 5%.

### u04 — Which programming language is recommended for writing smart contracts on a blockchain?

- **Đáp án chuẩn:** (câu thiếu evidence) Trang 393-394 chỉ nhắc blockchain khi nói về mã hóa của tiền mã hóa, không nói về smart contract hay ngôn ngữ lập trình.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch08:p393:3, B:pds-ch02:p80:1, B:pds-ch02:p77:1
- **Nhãn gợi ý:** answer_grade=None, citation_support=[], abstention_correct=True, context_grade=None
- **Lý do:** Từ chối, không bịa tên ngôn ngữ; corpus không nhắc smart contract.

## ag_base

### t01 — During data chunking, what metadata is added to each chunk so that the chunks can be reassembled into the original dataset?

- **Đáp án chuẩn:** The chunk number and the total number of chunks.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch02:p99:3, B:pds-ch02:p103:1
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context chỉ có p99:3, p103:1, không chứa evidence.

### t02 — Which external Python library is mainly used for scraping and parsing HTML and XML code?

- **Đáp án chuẩn:** Beautiful Soup.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** (rỗng)
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context rỗng.

### t03 — Which normal forms are most commonly used in database design?

- **Đáp án chuẩn:** 1NF, 2NF, 3NF (First, Second and Third Normal Form) and BCNF (Boyce-Codd Normal Form).
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** (rỗng)
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context rỗng.

### t04 — In the United States, how long does copyright protect original creative works?

- **Đáp án chuẩn:** For the life of the creator plus 70 years.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch08:p388:4
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context chỉ có p388:4, không chứa evidence.

### t05 — Which platform is mentioned for sharing data with the broader public in a research context when there are no privacy implications?

- **Đáp án chuẩn:** Kaggle, which is designed for open data sharing in a research context.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** (rỗng)
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context rỗng.

### t06 — In the plant growth experiment on sunlight, how many hours of sunlight per day did each of the two groups receive?

- **Đáp án chuẩn:** One group received eight hours of sunlight per day and the other only four hours.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** (rỗng)
- **Nhãn gợi ý:** answer_grade=0, citation_support=[], abstention_correct=None, context_grade=0
- **Lý do:** Từ chối câu có đáp án -> 0. Budget 800 ký tự nhỏ hơn độ dài chunk (~900) nên chunk đúng bị loại khỏi context; context rỗng.

### u03 — What sample size is needed for a survey to reach a 95% confidence level with a 5% margin of error?

- **Đáp án chuẩn:** (câu thiếu evidence) Mục 2.2 nói về sampling và sample size nói chung, không có công thức/số mẫu cho mức tin cậy 95%.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch02:p73:4
- **Nhãn gợi ý:** answer_grade=None, citation_support=[], abstention_correct=True, context_grade=None
- **Lý do:** Từ chối, không đưa khẳng định nào.

### u04 — Which programming language is recommended for writing smart contracts on a blockchain?

- **Đáp án chuẩn:** (câu thiếu evidence) Trang 393-394 chỉ nhắc blockchain khi nói về mã hóa của tiền mã hóa, không nói về smart contract hay ngôn ngữ lập trình.
- **Agent (abstain):** Không đủ bằng chứng trong tài liệu để trả lời.
- **Context cuối:** B:pds-ch02:p77:1
- **Nhãn gợi ý:** answer_grade=None, citation_support=[], abstention_correct=True, context_grade=None
- **Lý do:** Từ chối, không đưa khẳng định nào (dù agent có thử query 'Solidity', không có trong corpus).

