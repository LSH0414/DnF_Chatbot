# Dungeon & Figher Chatbot 프로젝트

<h3> Updates </h3>

<ul>
<h4> 24.05.02 Update </h4>

기존 임베딩 모델인 bge-m3에 던전앤파이터 QA 데이터를 학습시켜 임베딩 모델을 [DNF-bge-m3](https://huggingface.co/COCO0414/bge-m3_finetune_dnf)로 업데이트 하였습니다. DNF-bge-m3는 던전앤파이터 공식 홈페이지에 있는 질문 게시판 데이터를 기반으로 훈련된 데이터로 던전앤파이터와 관련 텍스트를 위한 임베딩 모델입니다.

</ul>

<ul>
<h4> 24.04.20 Update </h4>

기존에 진행했던 [polyglot기반 챗봇](https://github.com/LSH0414/Project/tree/master/DnF_Chatbot)은 RAG를 구현하기 어려워,  [EEVE-10.8b](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)에 SFT를 새롭게 진행하였습니다.

</ul>


---


모델의 훈련 데이터는 이전 RLHF-PPO 과정에서 사용한 데이터와 동일하나 챗봇 데이터로 변환해 미세조정하였습니다. EEVE모델은 한국어 중심 모델이며, RAG 프롬프트에 대응할 수 있는 모델입니다.

</br>
<h3>RAG로 활용되는 데이터</h3>

- [던전앤파이터 공식 홈페이지 커뮤니티 공략글](https://df.nexon.com/community/dnfboard/article/2760672?category=0)
- [던전앤파이터 나무위키](https://namu.wiki/w/던전앤파이터)


이후 RAG 데이터는 던전앤파이터 인게임 정보와 스토레에 관련된 모든 데이터를 추가해볼 예정입니다.

<h4>RAG Options</h4>

<ul>
  - Chunk : 3,000<br/>
  - Overlap : 250<br/>
  - Retriever : Ensembel Retriever<br/>
  - VectorStore : FAISS<br/>
  - Embedding Model : <a href = 'https://huggingface.co/BAAI/bge-m3'>BAAI/bge-m3</a> <br/>
  - Data</br>
  <ul>
  - 공식 홈페이지 글 : 156개</br>
  - 나무위키 페이지 : 679개
    </ul>
</ul>


실행환경 : Macbook Air M2



https://github.com/LSH0414/Project/assets/119479455/8cff1b25-b580-43a2-a714-f676ab7523f0



</br></br>
훈련 모델은 gguf 파일로 huggingface에 공유 예정입니다.

현재 임베딩 모델은 오픈소스로 더 많은 데이터로 RAG 수행시 낮은 포퍼먼스를 보일 수 있습니다. 따라서 던전앤파이터에 대한 데이터를 좀 더 수집하고 임베딩 모델에 학습시켜 이를 보완할 예정입니다.
