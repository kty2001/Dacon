# test = pd.read_csv('./data/test.csv')
# test_mfcc = get_mfcc_feature(test, False)
# test_dataset = CustomDataset(test_mfcc, None)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=CONFIG.BATCH_SIZE,
#     shuffle=False
# )

# def inference(model, test_loader, device):
#     model.to(device)
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for features in tqdm(iter(test_loader)):
#             features = features.float().to(device)
            
#             probs = model(features)

#             probs  = probs.cpu().detach().numpy()
#             predictions += probs.tolist()
#     return predictions

# preds = inference(infer_model, test_loader, device)

# submit = pd.read_csv('./sample_submission.csv')
# submit.iloc[:, 1:] = preds
# submit.head()
# submit.to_csv('./baseline_submit.csv', index=False)