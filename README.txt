In this project a Streamlit Application with AI Multi-agent workflows has been build in Python for Resume matching of candidates for a particular role
with Langchain, LangGraph and Llama3 model (Open Source LLM) through Groq. Also used Hugging Face Instruct Embeddings, Chroma (open-source vector database) and PyPDFLoader etc.. In the App, user only needs to upload the resume copy of the candidate (pdf) and also the job description (text file). 
Then they just need to click on "Match Resume" button. It will display the LangGraph multi-agent workflows with all the nodes and edges there
and also then it will execute it and show the result (whether the candidate is a good fit for the given role or not)
with detailed explanations. In fact, it will show the output messages from all the nodes (agents) so
that one can get a complete picture apart from the final judgment.  

Following are the nodes (AI agents) and their roles in the multi-agent workflows:

Resume agent: It will go through the resume and extract the exact candidate name
JD agent: It will go through the job description and extract the exact job requirements
RAG agent: It will find out the LinkedIn profile information of the candidate from the resume
Web scrap agent: It will extract the exact LinkedIn profile link (url) of the candidate 
                 from information sent by the previous node, then it will go to that url, 
                 web scraps the basic information like name and check if it is matching with 
                 the name given in resume.
Recruiter agent (final agent): It will collect the information from JD agent and Web scrap agent and 
                               it will also go through the candidate resume and then will make the final
                               judgment whether the candidate is a good fit for the role or not with detailed explanations.




