import torch
import re
from sklearn.metrics import f1_score
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings



# Очистка текста cleaner
# регуляризация l1 / l2
# f1 метрика для батча
# тренировочный цикл


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")
print('CUDA ?: ', torch.cuda.is_available())


def cleaner(text_line):
    clean_text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\.\,\!\?\:\;\"\'\(\)\-\s]', '', text_line)
    clean_text = re.sub(r'[\n\r\t\\]', '', clean_text)
    clean_text = clean_text.strip()
    return clean_text



def loss_with_l1_or_l2_regulizer(
        
        model,
        type, # Здесь могут быть 3 вида : 'l1', 'l2', 'l1_l2' 
        l_lambda, # 0.001
        loss_by_fn):
    
    if type == 'l2':
        l_norm = sum(p.pow(2.0).sum() for p in model.parameters())

    if type == 'l1':
        l_norm = sum(p.abs().sum() for p in model.parameters())

    if type == 'l1_l2':
        l_norm = sum((p.abs() + p.pow(2.0)).sum() for p in model.parameters())
    
    loss_plus_reg = loss_by_fn + l_lambda * l_norm
    return loss_plus_reg



def create_folder(name):
    import os
    os.mkdir(name)



def batch_metric(output, y_true):
    output = torch.argmax(output, dim = 1)
    output = output.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    f1_score_macro = f1_score(
        y_true = y_true,
        y_pred = output,
        average='macro'
    )
    return f1_score_macro




def training_loop(
        # папка сохранения
        name_model_folder,


        # основные настройки
        epoches,
        model,
        optimizer,

        # контроль скорости
        scheduler_append,
        scheduler,
        step_on, # str 'val_loss', 'train_loss'

        loss_fn,
        train_dataloader,
        val_dataloader,

        # regulizer
        regulizer_append,
        reg_type,
        
        # stop settings
        stop_mode,
        each__X__epoch,
        min_delta,
        patience,

        # verbose
        verbose_epoch
        ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    create_folder(name_model_folder)
    
    loss_train_list = []
    loss_val_list = []
    f1_train_list = []
    f1_val_list = []
    best_loss = float('inf')
    patience_beginner = patience

    model = model.to(device)

    for epoch in range(1, epoches+1):
            
            start_time = time.time()

            loss_train = 0.0
            loss_val = 0.0
            f1_train_epoch_sum = 0.0
            f1_val_epoch_sum = 0.0

            for batch in train_dataloader:
                
                model.train()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)[0]

                # Тренировочный потери
                loss = loss_fn(outputs, labels)
                
                if regulizer_append:
                        loss = loss_with_l1_or_l2_regulizer(
                                model = model,
                                type = reg_type,
                                l_lambda = 0.001,
                                loss_by_fn = loss
                                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss_train += loss.item()

                f1_train_epoch_sum += batch_metric(outputs, labels)
                

            mean_epoch_loss_train = loss_train/len(train_dataloader)
            loss_train_list.append(mean_epoch_loss_train)
            ######## Расчет метрики на тренировочном наборе #####
            f1_train_epoch = f1_train_epoch_sum / len(train_dataloader)
            f1_train_list.append(f1_train_epoch)
            #####################################################
            # Проверочный этап
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['labels'].to(device)
                        outputs = model(input_ids, attention_mask)[0]
                        loss_on_val = loss_fn(outputs, labels)
                        loss_val += loss_on_val.item()

                        f1_val_epoch_sum += batch_metric(outputs, labels)

                mean_epoch_loss_val = loss_val / len(val_dataloader)
                loss_val_list.append(mean_epoch_loss_val)

                ######## Расчет метрики на валидационном наборе #####
                f1_val_epoch = f1_val_epoch_sum / len(val_dataloader)
                f1_val_list.append(f1_val_epoch)
                ######################################################
                
            # EARLY STOPPING #######################################
            if stop_mode:
                if epoch % each__X__epoch == 0:
                        current_loss = mean_epoch_loss_train
                        if current_loss < best_loss - min_delta:
                                best_loss = current_loss
                                patience = patience_beginner
                        else:
                                patience -= 1
                                if patience == 0:
                                        print(f"Ранняя тренировочная остановка на {epoch} эпохе")
                                        break
            ##############################################################
            # Сохранение состояний модели
            if mean_epoch_loss_val < best_loss:
                    best_loss = mean_epoch_loss_val
                    torch.save(model.state_dict(), f'{name_model_folder}/model_weights_{epoch}_epoch.pth')
            # Контроль скорости обучения
            if scheduler_append:
                if step_on == 'val_loss':
                        scheduler.step(mean_epoch_loss_val)
                if step_on == 'train_loss':
                        scheduler.step(mean_epoch_loss_train)
            ##############################################################
            end_time = time.time()
            epoch_time = end_time - start_time
            расчетное_время_выполнения_цикла = (epoch_time*epoches) / 60
            ############ VERBOSE #########################################
            if epoch ==1:
                   print('Расчетное время выполнения всего цикла {:.3f} минут'.format(расчетное_время_выполнения_цикла))            
            if epoch == 1 or epoch % verbose_epoch == 0:
                print(
                        'Epoch {:5}/{:5} || time: {:.3f} || train loss: {:.3f} || val_loss: {:.3f} || train f1: {:.3f} || val f1: {:.3f}'
                        .format(epoch,
                                epoches,
                                epoch_time,
                                mean_epoch_loss_train,
                                mean_epoch_loss_val,
                                f1_train_epoch,
                                f1_val_epoch,
                                )     
                      )
    ###### PLOTTING #############################################################################
    sns.lineplot(x = np.arange(1, epoch+1),
                 y = loss_train_list,
                 label = 'train')
    sns.lineplot(x = np.arange(1, epoch+1),
                 y = loss_val_list,
                 label = 'val')
    plt.title('Потери от эпох')
    plt.xlabel('эпохи')
    plt.ylabel('loss')
    plt.legend()
    plt.grid()
    plt.show()

    sns.lineplot(x = np.arange(1, epoch+1),
                 y = f1_train_list,
                 label = 'train')
    sns.lineplot(x = np.arange(1, epoch+1),
                 y = f1_val_list,
                 label = 'val')
    
    plt.xlabel('эпохи')
    plt.ylabel('accuracy')
    plt.title('accuracy on train: {}   lr: {}'.format(epoches,
                                                      optimizer.param_groups[0]['lr']))
    plt.legend()
    plt.grid()
    plt.show()