def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for users, items, ratings in dataloader:
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            outputs = model(users, items)
            total_loss += criterion(outputs, ratings).item()
    return (total_loss / len(dataloader)) ** 0.5  # Convert MSE to RMSE

test_rmse = evaluate(model, test_loader, nn.MSELoss(), device)
print(f'Test RMSE: {test_rmse:.4f}')
