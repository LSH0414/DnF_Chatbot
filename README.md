# Dungeon & Figher Chatbot 프로젝트

<h3> Updates </h3>

<ul>
<h4> [24.05.02 Update] </h4>

기존 임베딩 모델인 bge-m3에 던전앤파이터 QA 데이터를 학습시켜 임베딩 모델을 [DNF-bge-m3](https://huggingface.co/COCO0414/bge-m3_finetune_dnf)로 업데이트 하였습니다. DNF-bge-m3는 던전앤파이터 공식 홈페이지에 있는 질문 게시판 데이터를 기반으로 훈련된 데이터로, 던전앤파이터와 관련 있는 텍스트를 잘 표현하는 임베딩 모델입니다.

</ul>

<ul>
<h4> [24.04.20 Update] </h4>

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



<br><br>


<h2>DNF-bge-m3</h2>

bge-m3을 베이스 모델로 2만개의 질문글을 바탕으로 7만 9천개의 QA pairs 데이터를 만들어 추가 훈련 시킨 임베딩 모델입니다. 훈련은 3-epochs 진행하였고 3회부터 변별력에 큰 차이가 발생하지 않는 것을 확인할 수 있었습니다.

첫 훈련이 베이스 모델과 가장 차이가 큰 유사도 결과를 얻었고 훈련을 거듭할수록 안정적인 모습을 확인할 수 있었습니다. 또한, 훈련을 거듭할 수록 유사도 점수 자체는 낮아질 수 있으나 정확한 필터링이 이뤄지고 있는 모습을 확인할 수 있었습니다. 실제로 임계값(0.6, 0.7)을 정하여 확인해본 결과 2-epoch에서 타겟 document와 유사도가 가장 높은 결과를 확인했으나 다른 5개 document와도 높은 유사도를 갖은 반면 3-epoch에서는 타겟 document와 함께 1개의 추가 document가 필터링 되었습니다. 이는 RAG 성능과 포퍼먼스에 크게 영향을 줄 수 있어 3-epoch모델을 최종 모델로 선정하였습니다.

베이스 모델이 평균 유사도가 가장 높게 나타났지만 어떠한 질문에도 비슷한 유사도를 반환해 변별력이 없었습니다. 그에 반면 훈련된 임베딩 모델은 이전과 다르게 관련이 없는 문서에 대해서는 낮은 유사도 결과를 보여주는 것을 확인하였습니다. 전체적인 유사도 점수가 낮아진 것은 좀 더 변별력을 갖춘 모델이 되었다는 것을 의미합니다.

</br>

<h3>Compare origin model(bge-m3)</h3>

<h4> Cosin Similarity Mean</h4>

| Epoch         | Value  |
|---------------|--------|
| Origin        | 0.5558 |
| 1-epoch       | 0.4099 |
| 2-epoch       | 0.4772 |
| 3-epoch       | 0.4556 |


<h4> Cosin Similarity Hist</h4>
<ul>
  <h5>1-epoch</h5>
  <img src = 'https://github.com/LSH0414/DnF_Chatbot/assets/119479455/779ac4e5-426e-4e0f-a255-6ed23613e965' width="600"/>

  <h5>2-epoch</h5>
  <img src = 'https://github.com/LSH0414/DnF_Chatbot/assets/119479455/4c5be840-b9ee-4c3b-9b8b-6ce08fd99428' width="600"/>

  <h5>3-epoch</h5>
  <img src = 'https://github.com/LSH0414/DnF_Chatbot/assets/119479455/2232db9b-3ea9-4cfd-8f79-f74b094a9adb' width="600"/>

</ul>





</br></br>

---

[Note]

훈련 모델은 gguf 파일로 huggingface에 공유 예정입니다.

