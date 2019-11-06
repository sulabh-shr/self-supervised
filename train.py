import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from metric_learning_data import ActiveVisionTriplet
from metric_learning import MetricLearningNet

EPOCH = 1
BATCH_SIZE = 15
PRINT_EVERY = 50
VAL_EVERY = PRINT_EVERY * 3
VAL_SPLIT = 0.1
VAL_SAMPLE = 80  # Batch size for validation. Performs for only 1 batch
LEARNING_RATE = 0.0001

model = MetricLearningNet()
model = model.cuda()
criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
total_dataset = ActiveVisionTriplet('/mnt/sda2/workspace/DATASETS/ActiveVision',
                                    '/home/sulabh/workspace-ubuntu/triplets',
                                    instance='instance1',
                                    image_size=(1333, 750),
                                    triplet_image_size=(224, 224))

print(f'Total number of examples = {len(total_dataset)}')
val_length = round(len(total_dataset)*0.1)
train_length = len(total_dataset) - val_length
print(f'Train length = {train_length}\nVal length = {val_length}')
print(f'Validation Sample = {VAL_SAMPLE}')

train_dataset, val_dataset = random_split(total_dataset, [train_length, val_length])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=VAL_SAMPLE, num_workers=4)

running_loss = 0.0

for epoch in range(EPOCH):
    print(f'Epoch = {epoch + 1}')

    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        ref, pos, neg = data
        ref = ref.cuda()
        pos = pos.cuda()
        neg = neg.cuda()

        optimizer.zero_grad()

        ref_emb, pos_emb, neg_emb = model(ref, pos, neg)
        loss = criterion(ref_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if idx % PRINT_EVERY == 0 or idx % VAL_EVERY == 0:

            if idx % VAL_EVERY == 0:
                with torch.no_grad():

                    for val_data in val_loader:
                        val_ref, val_pos, val_neg = val_data
                        val_ref = val_ref.cuda()
                        val_pos = val_pos.cuda()
                        val_neg = val_neg.cuda()

                        out = model(val_ref, val_pos, val_neg)
                        val_loss = criterion(*out)
                        break

                    print(f'Iteration: {idx}, loss = {running_loss/PRINT_EVERY:.5f}, '
                          f'val_loss = {val_loss:.5f}')

            else:
                print(f'Iteration: {idx}, loss = {running_loss/PRINT_EVERY:.5f}')

            running_loss = 0.0






