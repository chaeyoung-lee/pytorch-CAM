import torch
from torch.autograd import Variable

def retrain(trainloader, model, use_cuda, epoch, criterion, optimizer):
    model.train()
    correct, total = 0, 0
    acc_sum, loss_sum = 0, 0
    i = 0
    for batch_idx, (data, target) in enumerate(trainloader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        # calculate accuracy
        correct += (torch.max(output, 1)[1].view(target.size()).data == target.data).sum()
        total += trainloader.batch_size
        train_acc = 100. * correct / total
        acc_sum += train_acc
        i += 1

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}\tTraining Accuracy: {:.3f}%'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item(), train_acc))

    acc_avg = acc_sum / i
    loss_avg = loss_sum / len(trainloader.dataset)
    print()
    print('Train Epoch: {}\tAverage Loss: {:.3f}\tAverage Accuracy: {:.3f}%'.format(epoch, loss_avg, acc_avg))

    with open('result/train_acc.txt', 'a') as f:
        f.write(str(acc_avg))
    f.close()
    with open('result/train_loss.txt', 'a') as f:
        f.write(str(loss_avg))
    f.close()

def retest(testloader, model, use_cuda, criterion, epoch, RESUME):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(testloader.dataset)
    test_acc = 100. * correct / len(testloader.dataset)
    result = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(testloader.dataset), test_acc)
    print(result)

    # Save checkpoint.
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'checkpoint/' + str(RESUME + int(epoch / 10)) + '.pt')
        with open('result/result.txt', 'a') as f:
            f.write(result)
        f.close()

    with open('result/test_acc.txt', 'a') as f:
        f.write(str(test_acc))
    f.close()
    with open('result/test_loss.txt', 'a') as f:
        f.write(str(test_loss))
    f.close()
