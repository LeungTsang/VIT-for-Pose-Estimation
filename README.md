# pose_vit

Use transformer to regress poses between multiple frames in one shot.  
Transfomer is efficient for processing video by joint space-time attention. (Is Space-Time Attention All You Need for Video Understanding?
https://arxiv.org/pdf/2102.05095)  
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/pic1.png" width="160px">
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/pic2.png" width="160px">
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/pic3.png" width="160px">

Network Architecture  
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/Architecture_pose_vit.png" width="300px">  

RMSE to GT trajectory in KITTI. Performance increases when more frames are involved.  
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/rmse9.png" width="200px">
<img src="https://github.com/LeungTsang/pose_vit/raw/master/fig/rmse10.png" width="200px">
