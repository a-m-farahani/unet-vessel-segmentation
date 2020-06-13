# Retinal Blood Vessel Segmentation 
Retinal blood vessel segmentation using a tiny U-Net model with Adaptive Activation Functions.

<br/>
<b> Tiny U-Net model architecture.</b>
<p>
  <img src="https://github.com/a-m-farahani/unet-vessel-segmentation/blob/master/images/TinyUnet_model.png" height="250" title="Tiny U-Net architecture">
</p>

In this model each convolution layer has its own activation function that is a linear combination of 14 base activation functions.
<br/>
Different Adaptive Activation Functions of the model trained on DRIVE dataset.
<p>
  <img src="https://github.com/a-m-farahani/unet-vessel-segmentation/blob/master/images/AAFx14_DRIVE.png" height="200" title="14 AAF">
</p>

<br/>
<b>Results:</b>
<p>
  <img src="https://github.com/a-m-farahani/unet-vessel-segmentation/blob/master/images/chasedb_test.png" height="200" title="Vessel Segmentation Result on CHASEDB dataset">
</p>
