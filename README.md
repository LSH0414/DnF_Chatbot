# Dungeon & Figher Chatbot 프로젝트


<h3> 24.04.20 Update</h3>

기존에 진행했던 [polyglot기반 챗봇](https://github.com/LSH0414/Project/tree/master/DnF_Chatbot)은 RAG를 구현하기 어려워, QA데이터를 기반으로 [EEVE-10.8b](https://huggingface.co/yanolja/EEVE-Korean-Instruct-10.8B-v1.0)에 SFT를 진행하였습니다.

RAG로 활용되는 데이터는 아래 두 페이지에 있는 글을 수집하여 진행하였습니다.
- [던전앤파이터 공식 홈페이지 커뮤니티 공략글](https://df.nexon.com/community/dnfboard/article/2760672?category=0)
- [던전앤파이터 나무위키](https://namu.wiki/w/던전앤파이터)

수집한 데이터를 통해 추가적으로 임베딩에 대한 학습 진행예정입니다.
모델은 gguf 파일로 huggingface에 공유 예정입니다.

<h4>RAG Options</h4>

<ul>
  - Chunk : 3,000<br/>
  - Overlap : 250<br/>
  - Embedding Model : BAAI/bge-m3<br/>
  - Data</br>
  <ul>
  - 공식 홈페이지 글 : 156개</br>
  - 나무위키 페이지 : 679개
    </ul>
</ul>


실행환경 : Macbook Air M2



https://github.com/LSH0414/Project/assets/119479455/8cff1b25-b580-43a2-a714-f676ab7523f0


