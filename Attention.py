#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalAttention(nn.Module):
    def __init__(self, input_dim):
        super(MultimodalAttention, self).__init__()
        # 线性变换
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.fc3 = nn.Linear(input_dim, input_dim)

    def forward(self, text, audio, video):
       
        text_audio = torch.tanh(self.fc1(text + audio))
        text_video = torch.tanh(self.fc2(text + video))
        audio_video = torch.tanh(self.fc3(audio + video))

        
        combined_features = text_audio + text_video + audio_video

        # 计算注意力权重
        attention_weights = F.softmax(combined_features, dim=1)

        # 将注意力权重应用到每个模态
        text_weighted = attention_weights * text
        audio_weighted = attention_weights * audio
        video_weighted = attention_weights * video

        return text_weighted + audio_weighted + video_weighted


# In[ ]:




