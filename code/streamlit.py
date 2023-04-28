import streamlit as st
from dataloader import Dataloader
from arguments import get_args
import pandas as pd
import pytorch_lightning as pl
import torch

# Set page config to wide mode
st.set_page_config(layout='wide')

# Title
st.title("Semantic Text Similarity")
st.write("두 문장의 의미적 유사도를 측정해보자!")
st.write("아래에 두 문장을 입력해주세요.")
# Balloon effect
st.balloons()

# Text input
text_input1 = st.text_input("Sentence1")
text_input2 = st.text_input("Sentence2")

with open('./code/data/test.csv', 'w', encoding='UTF8') as f:
    f.write(f'id,source,sentence_1,sentence_2\n0,abc,{text_input1},{text_input2}')

# 실행 버튼 클릭
if st.button("실행"):
    with st.spinner("Please wait.."):
        args = get_args()
        dataloader = Dataloader(args.model_name, args.batch_size, args.shuffle, args.train_path, args.dev_path,
                                args.test_path, args.predict_path)
        trainer = pl.Trainer(accelerator='cpu', max_epochs=args.max_epoch, log_every_n_steps=1)

        # Inference part
        # 저장된 모델로 예측을 진행합니다.
        model = torch.load('./code/model_robertaLarge_2epochs_stepLR.pt')
        predictions = trainer.predict(model=model, datamodule=dataloader)
    
    st.write(round(float(predictions[0]), 1) * 20)
