## SFT vs DPO

In general first SFT is done and afterwords DPO is done. SFT works by giving the model reference examples. DPO works by giving the model a good and bad answer.
</br>

According to [youtube-vid](https://www.youtube.com/watch?v=E5kzAbD8D0w&list=PLWG1mVtuzdxfXkxCbPHh9reKV-fWqraEX&index=15) SFT increases the probability of good answers by simply adding more good answers to the models probabilty distribution. So with SFT we increase the frequency of good answers. While DPO moves the probability distribution of the model towards the good answers and away from the bad answers. So we can get actually get rid of bad answers.
</br>

Your LLM is also pretrained on lots of bad data and only with DPO we can move the model away from it.