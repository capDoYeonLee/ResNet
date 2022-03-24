from train import * 
from model import *



correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        images, labels = images.to(device), labels.to(device)

        # calculate outputs by running images through the network
        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    # print('correct : ', correct)
    # print('total : ', total)

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')





def show_GT_and_PD():  
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    # Ground Truth
    # imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))


    images, labels = images.cuda(), labels.cuda()
    images, labels = images.to(device), labels.to(device)

    # Predicted output
    outputs = model(images)    
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))
    
    return 