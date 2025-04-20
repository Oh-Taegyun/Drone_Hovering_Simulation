# Drone_Hovering_Simulation

드론 호버링 시뮬레이션을 SAC로 해본 결과를 올려보는 공간

- simulatior : Genesis-simulatior
- langage : python
- DeepLearning Framework : Pytorch

학습중인 영상

https://github.com/user-attachments/assets/6c80a2e5-61aa-4d57-90c3-bd60851ce164


학습이 완료 된 이후

단, 원래 코드에서 보상 함수를 보면 특정 지점에서 멀어질 경우에 새롭게 목표 지점이 생성되는 로직입니다. 

아래 영상은 제가 조금 개조해서, 목표 지점에 닿으면 새로운 목표 지점을 생성하도록 만들었습니다. 

원해는 학습이 다 되면 특정 지점에서 호버링 유지하는게 맞습니다. 


https://github.com/user-attachments/assets/7b10b222-52c4-4999-8cb0-7d1989536dbc

