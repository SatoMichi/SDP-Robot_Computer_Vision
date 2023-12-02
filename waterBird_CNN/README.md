# waterbird_cnn.py

Training will took time without using GPU(cuda).

If you want to add data, add to ./images/train(Ture for bird image, False for no-bird image)

If you just want to test it, you can comment out the following part:
"""

    start = time.time()
    trainModel(net,train_loader,criterion,optimizer,100)
    process_time = time.time() - start
    print(process_time)
    
    PATH = './waterBird_net.pth'
    torch.save(net.state_dict(), PATH)
"""

and load the parameter from './waterBird_net_1.pth'

to show the prediction result predict(model,path): function can be used.

model: in this code it is net(WaterBirdCNNet) object.
path: image path (e.g. "/.bird_yes.jpg")


