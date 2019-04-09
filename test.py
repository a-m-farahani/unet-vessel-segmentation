#model.load_state_dict(torch.load("gdrive/My Drive/models/drive_aaf14x16_exp1.mdl"))

n_test = dataset.Test.n
#dataset.Test.reset()
sum_of_losses = 0
TPR = [0] * (n_test+1)
TNR = [0] * (n_test+1)
ACC = [0] * (n_test+1)
for i in range(n_test):
  img,seg,fname = dataset.Test.next_image()
  print(fname)
  img = normalizer(transforms.functional.to_tensor(img)).unsqueeze(0).cuda()
  
  y_pred = model(img)
  
  seg_ = transforms.functional.to_tensor(seg).cuda().unsqueeze(0)
  loss = loss_function(y_pred,seg_)
  temp = confusion(y_pred,seg_)
  TPR[i+1] = temp["TPR"]
  TNR[i+1] = temp["TNR"]
  ACC[i+1] = temp["Accuracy"]
  TPR[0] += temp["TPR"]/n_test
  TNR[0] += temp["TNR"]/n_test
  ACC[0] += temp["Accuracy"]/n_test
  print(temp)
  sum_of_losses += loss.item()
  
  tmp = transforms.functional.to_pil_image(y_pred.cpu().squeeze(0))
  #tmp.save("gdrive/My Drive/DataSets/SegRes/"+fname,"GIF")
  
  plt.subplot(1,3,1)
  plt.imshow(transforms.functional.to_pil_image(img.cpu().squeeze(0)))
  plt.subplot(1,3,2)
  plt.imshow(seg)
  plt.subplot(1,3,3)
  plt.imshow(transforms.functional.to_pil_image(y_pred.cpu().squeeze(0)))
  plt.show()
  print("Test Case {}, IoU = {:.4f}".format(i+1,loss.item()))
  time.sleep(1)
  
print("Average Loss over {} test casees is {:.4f}".format(n_test,sum_of_losses/n_test))
print("Sensitivity = {:.4f}, Specificity = {:.4f}, Accurcay = {:.4f}".format(TPR[0],TNR[0],ACC[0]))
