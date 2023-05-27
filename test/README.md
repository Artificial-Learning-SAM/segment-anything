所有代码在本文件夹内。

以下的命令为选择三个点作为prompt，其中第一个是内距离最大点：
```
cd test
python task1.py -h
python task1.py -n 3 -c
```

额外需要的文件（不添加到仓库内）结构：
```
segment-anything/
├─ test/
│  ├─ BTCV
│  │  ├─ imagesTr
|  │  │  ├─ img0001.nii.gz
│  │  ├─ labelsTr
|  │  │  ├─ label0001.nii.gz
├─ sam_vit_h_4b8939.pth
```