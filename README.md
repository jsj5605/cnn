# cnn

## 1. cnn 모델 정의
```python
# CNN -> Convolution Layer: fitlter 개수(out_channels로 설정) 뒤로 갈수록 크게 잡는다.
# Max Pooling Layer를 이용해서 출력 결과(Feature map)의 
# size(height, width) 는 줄여나간다. (보통 절반씩 줄인다.)

# conv block
## 1. Conv + ReLU + MaxPooling
## 2. Conv + BatchNorm + ReLU + MaxPooling
## 3. Conv + BatchNorm + ReLU + Dropout + MaxPooling


class MNISTCNNModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            # Conv2d(): 3 X 3 필터, stide=1, padding=1 => same padding(입력 size와 출력 size가 동일)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), # channel을 기준을 정규화 -> 입력 channel수 지정.
            nn.ReLU(), 
            nn.Dropout2d(p=0.3), 
            nn.MaxPool2d(kernel_size=2, stride=2)
            # kernel_size와 stride가 같은 경우에는 stride를 생략가능
            # MaxPool2d() 에서도 padding 지정.
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2, padding=1)  # 입력: 7 x 7 => 1/2 줄이면 -> 3.5 -> 0.5를 살리기 위해 padding 지정
        )
        
        # 결과출력레이어 => Linear() 사용. 
        self.output_block = nn.Sequential(
            # MaxPool2d() 출력결과 입력으로 받는다. => 4차원 (batch, ch, h, w)
            # 3차원 -> 1차원
            nn.Flatten(), 
            nn.Linear(in_features=128*4*4, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 10) # out=>클래스 개수
        )
        
    def forward(self, X):
        out = self.b1(X)
        out = self.b2(out)
        out = self.b3(out)
        out = self.output_block(out)
        
        return out
```
