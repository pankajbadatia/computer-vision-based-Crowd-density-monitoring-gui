"""
Main code for a GAN training session.
"""
import datetime
import os
import torch.utils.data
import torchvision
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import lr_scheduler, Adam
from settings import Settings
import transforms
import viewer
from crowd_dataset import CrowdDataset
from hardware import gpu, cpu
from model import Generator, Discriminator1, Discriminator2, Discriminator3, load_trainer, save_trainer
#from settings import Settings

settings = Settings()
settings.train_dataset_path = '/media/pankaj/2E301C9E301C6ED9/Users/Pankaj/Downloads/pank do not throw/1 Camera 1 Images Target Unlabeled/test'
settings.batch_size = 400

################################### Do image transformation ############################################

train_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.NegativeOneToOneNormalizeImage(),
                                                  transforms.NumpyArraysToTorchTensors()])
validation_transform = torchvision.transforms.Compose([transforms.RandomlySelectPatchAndRescale(),
                                                       transforms.NegativeOneToOneNormalizeImage(),
                                                       transforms.NumpyArraysToTorchTensors()])

################################## Load Data from  path ##################################################

train_dataset = CrowdDataset(settings.train_dataset_path, 'train', transform=train_transform)
train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True,num_workers=settings.number_of_data_loader_workers)
validation_dataset = CrowdDataset(settings.validation_dataset_path, 'validation', transform=validation_transform)
validation_dataset_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=settings.batch_size, shuffle=False, num_workers=settings.number_of_data_loader_workers)


################################## Create Objects of models ###################################################
generator = Generator()

discriminator1 = Discriminator1()
discriminator2 = Discriminator2()
discriminator3 = Discriminator3()

gpu(generator)

gpu(discriminator1)
gpu(discriminator2)
gpu(discriminator3)

################################## Create Optimizer objects ##################################################

generator_optimizer = Adam(generator.parameters())
discriminator_optimizer1 = Adam(discriminator1.parameters())
discriminator_optimizer2 = Adam(discriminator2.parameters())
discriminator_optimizer3 = Adam(discriminator3.parameters())


################ Initialize Variables###########################
step = 0
epoch = 0

###################################### Loading for Discriminator #####################################

if settings.load_model_path:
    d_model_state_dict, d_optimizer_state_dict, epoch, step = load_trainer(prefix='discriminator1')

    discriminator1.load_state_dict(d_model_state_dict)
    discriminator_optimizer1.load_state_dict(d_optimizer_state_dict)
    
    discriminator2.load_state_dict(d_model_state_dict)
    discriminator_optimizer2.load_state_dict(d_optimizer_state_dict)
    
    discriminator3.load_state_dict(d_model_state_dict)
    discriminator_optimizer3.load_state_dict(d_optimizer_state_dict)

discriminator_optimizer1.param_groups[0].update({'lr': settings.initial_learning_rate,'weight_decay': settings.weight_decay})
discriminator_scheduler1 = lr_scheduler.LambdaLR(discriminator_optimizer1,lr_lambda=settings.learning_rate_multiplier_function)
discriminator_scheduler1.step(epoch)


discriminator_optimizer2.param_groups[0].update({'lr': settings.initial_learning_rate,'weight_decay': settings.weight_decay})
discriminator_scheduler2 = lr_scheduler.LambdaLR(discriminator_optimizer2,lr_lambda=settings.learning_rate_multiplier_function)
discriminator_scheduler2.step(epoch)


discriminator_optimizer3.param_groups[0].update({'lr': settings.initial_learning_rate,'weight_decay': settings.weight_decay})
discriminator_scheduler3 = lr_scheduler.LambdaLR(discriminator_optimizer3,lr_lambda=settings.learning_rate_multiplier_function)
discriminator_scheduler3.step(epoch)

########################################  Loading for Generator  #################################################################

if settings.load_model_path:
    g_model_state_dict, g_optimizer_state_dict, _, _ = load_trainer(prefix='generator')
    generator.load_state_dict(g_model_state_dict)
    generator_optimizer.load_state_dict(g_optimizer_state_dict)

generator_optimizer.param_groups[0].update({'lr': settings.initial_learning_rate})
generator_scheduler = lr_scheduler.LambdaLR(generator_optimizer, lr_lambda=settings.learning_rate_multiplier_function)
generator_scheduler.step(epoch)

#####################################  Code for Tensorboard  ##########################################
running_scalars = defaultdict(float)
validation_running_scalars = defaultdict(float)

running_example_count = 0

########### Creating directory  Log path , summary writer for train and validation #################
datetime_string = datetime.datetime.now().strftime("y%Ym%md%dh%Hm%Ms%S")
trial_directory = os.path.join(settings.log_directory, settings.trial_name + ' ' + datetime_string)
os.makedirs(trial_directory, exist_ok=True)
summary_writer = SummaryWriter(os.path.join(trial_directory, 'train'))
validation_summary_writer = SummaryWriter(os.path.join(trial_directory, 'validation'))

######################################### Start Training   ###############################################
print('Starting training...')
while epoch < settings.number_of_epochs:
    for examples in train_dataset_loader:

################# Real image discriminator1 processing #################################
        images, labels, _ = examples
        images, labels = Variable(gpu(images)), Variable(gpu(labels))

        discriminator_optimizer1.zero_grad()

        predicted_labels1, predicted_counts1 = discriminator1(images) # discriminator (input = real image = images)

        density_loss1 = torch.abs(predicted_labels1 - labels).pow(settings.loss_order).sum(1).sum(1).mean()
        count_loss1 = torch.abs(predicted_counts1 - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
        loss1 = count_loss1 + (density_loss1 * 10)
        print('loss 1 = ', loss1.shape)
        
        #loss1.backward()

################# Real image discriminator2 processing #################################
        discriminator_optimizer2.zero_grad()

        predicted_labels2, predicted_counts2 = discriminator2(images) # discriminator (input = real image = images)

        density_loss2 = torch.abs(predicted_labels2 - labels).pow(settings.loss_order).sum(1).sum(1).mean()
        count_loss2 = torch.abs(predicted_counts2 - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
        loss2 = count_loss2 + (density_loss2 * 10)
        print('loss 2 = ', loss2.shape)
        
        #loss2.backward()

################# Real image discriminator3 processing #################################
        discriminator_optimizer3.zero_grad()
        
        predicted_labels3, predicted_counts3 = discriminator3(images) # discriminator (input = real image = images)

        density_loss3 = torch.abs(predicted_labels3 - labels).pow(settings.loss_order).sum(1).sum(1).mean()
        count_loss3 = torch.abs(predicted_counts3 - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
        loss3 = count_loss3 + (density_loss3 * 10)
        print('loss 3 = ', loss3.shape)
        
        #loss3.backward()
        print('loss 1 =',loss1)
        print('loss 2 =',loss2)
        print('loss 3 =',loss3)
        totalloss = [loss1,loss2,loss3]
        diffloss = [loss1,loss2,loss3]
        lowloss = np.argmin(totalloss)
        print('loss index =',lowloss)
        print('min =',min(totalloss))
        
        diffloss[:] = [x - min(totalloss) for x in diffloss]
        print('diffloss = ', diffloss)
        comploss = [torch.abs(totalloss + diffloss) for totalloss, diffloss in zip(totalloss, diffloss)]
        
        print('comploss =', comploss[1])
        loss1 = comploss[0]
        loss1.backward(retain_graph=True)
        
        loss2 = comploss[1]
        loss2.backward(retain_graph=True)

        loss3 = comploss[2]
        loss3.backward(retain_graph=True)
        
        

################# Fake image discriminator1 processing #################################
        current_batch_size = images.data.shape[0]
        z = torch.randn(current_batch_size, 100)

        fake_images = generator(Variable(gpu(z)))# generator (input = random z )
        fake_predicted_labels1, fake_predicted_counts1 = discriminator1(fake_images) # discriminator (input = fake images)

        fake_density_loss1 = torch.abs(fake_predicted_labels1).pow(settings.loss_order).sum(1).sum(1).mean()
        fake_count_loss1 = torch.abs(fake_predicted_counts1).pow(settings.loss_order).mean()
        fake_discriminator_loss1 = fake_count_loss1 + (fake_density_loss1 * 10)
        
        fake_discriminator_loss1.backward(retain_graph=True)


################# Fake image discriminator2 processing #################################

        fake_predicted_labels2, fake_predicted_counts2 = discriminator2(fake_images) # discriminator (input = fake images)

        fake_density_loss2 = torch.abs(fake_predicted_labels2).pow(settings.loss_order).sum(1).sum(1).mean()
        fake_count_loss2 = torch.abs(fake_predicted_counts2).pow(settings.loss_order).mean()
        fake_discriminator_loss2 = fake_count_loss2 + (fake_density_loss2 * 10)
        
        fake_discriminator_loss2.backward(retain_graph=True)


################# Fake image discriminator3 processing #################################

        fake_predicted_labels3, fake_predicted_counts3 = discriminator3(fake_images) # discriminator (input = fake images)

        fake_density_loss3 = torch.abs(fake_predicted_labels3).pow(settings.loss_order).sum(1).sum(1).mean()
        fake_count_loss3 = torch.abs(fake_predicted_counts3).pow(settings.loss_order).mean()
        fake_discriminator_loss3 = fake_count_loss3 + (fake_density_loss3 * 10)
        
        fake_discriminator_loss3.backward(retain_graph=True)


#################################### Gradient penalty Discriminator 1 ##############################################
        alpha = Variable(gpu(torch.rand(current_batch_size, 1, 1, 1)))
        interpolates = alpha * images + ((1.0 - alpha) * fake_images)
        interpolates_labels1, interpolates_counts1 = discriminator1(interpolates) # discriminator(input = interpolated (real images + fake images) )

        density_gradients1 = torch.autograd.grad(outputs=interpolates_labels1, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_labels1.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        density_gradients1 = density_gradients1.view(current_batch_size, -1)
        density_gradient_penalty1 = ((density_gradients1.norm(2, dim=1) - 1) ** 2).mean() * 10

        count_gradients1 = torch.autograd.grad(outputs=interpolates_counts1, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_counts1.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        count_gradients1 = count_gradients1.view(current_batch_size, -1)
        count_gradients_penalty1 = ((count_gradients1.norm(2, dim=1) - 1) ** 2).mean() * 10

        gradient_penalty1 = count_gradients_penalty1 + density_gradient_penalty1 * 10
        




#################################### Gradient penalty Discriminator 2 ##############################################

        interpolates_labels2, interpolates_counts2 = discriminator2(interpolates) # discriminator(input = interpolated (real images + fake images) )

        density_gradients2 = torch.autograd.grad(outputs=interpolates_labels2, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_labels2.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        density_gradients2 = density_gradients2.view(current_batch_size, -1)
        density_gradient_penalty2 = ((density_gradients2.norm(2, dim=1) - 1) ** 2).mean() * 10

        count_gradients2 = torch.autograd.grad(outputs=interpolates_counts2, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_counts2.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        count_gradients2 = count_gradients2.view(current_batch_size, -1)
        count_gradients_penalty2 = ((count_gradients2.norm(2, dim=1) - 1) ** 2).mean() * 10

        gradient_penalty2 = count_gradients_penalty2 + density_gradient_penalty2 * 10
        
        



#################################### Gradient penalty Discriminator 3 ##############################################

        interpolates_labels3, interpolates_counts3 = discriminator3(interpolates) # discriminator(input = interpolated (real images + fake images) )

        density_gradients3 = torch.autograd.grad(outputs=interpolates_labels3, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_labels3.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        density_gradients3 = density_gradients3.view(current_batch_size, -1)
        density_gradient_penalty3 = ((density_gradients3.norm(2, dim=1) - 1) ** 2).mean() * 10

        count_gradients3 = torch.autograd.grad(outputs=interpolates_counts3, inputs=interpolates,grad_outputs=gpu(torch.ones(interpolates_counts3.size())),create_graph=True, retain_graph=True, only_inputs=True)[0]
        count_gradients3 = count_gradients3.view(current_batch_size, -1)
        count_gradients_penalty3 = ((count_gradients3.norm(2, dim=1) - 1) ** 2).mean() * 10

        gradient_penalty3 = count_gradients_penalty3 + density_gradient_penalty3 * 10
        


        
        print('gradient_penalty 1 =',gradient_penalty1)
        print('gradient_penalty 2 =',gradient_penalty2)
        print('gradient_penalty 3 =',gradient_penalty3)

        totalgradinetloss = [gradient_penalty1,gradient_penalty2,gradient_penalty3]
        
        diffgradientloss =  [gradient_penalty1,gradient_penalty2,gradient_penalty3]
        
        lowgradientloss = np.argmin(totalgradinetloss)
        print('loss index =',lowgradientloss)
        print('min =',min(totalgradinetloss))
        
        diffgradientloss[:] = [x - min(totalgradinetloss) for x in diffgradientloss]
        print('diffloss = ', diffgradientloss)
        gradcomploss = [torch.abs(totalgradinetloss + diffgradientloss) for totalgradinetloss, diffgradientloss in zip(totalgradinetloss, diffgradientloss)]
        
        print('comploss =', gradcomploss[1])
        gradient_penalty1 = gradcomploss[0]
        gradient_penalty1.backward(retain_graph=True)
        
        gradient_penalty2 = gradcomploss[1]
        gradient_penalty2.backward(retain_graph=True)

        gradient_penalty3 = gradcomploss[2]
        gradient_penalty3.backward(retain_graph=True)
        
        

############# Discriminator update ############
        
        discriminator_optimizer1.step()
        discriminator_optimizer2.step()
        discriminator_optimizer3.step()
        
######################################## Generator image processing #############################
        generator_optimizer.zero_grad()
        z = torch.randn(current_batch_size, 100)
        fake_images = generator(Variable(gpu(z))) # generator (input = random z)
        fake_predicted_labels1, fake_predicted_counts1 = discriminator1(fake_images) # discriminator(input = fake images)
        fake_predicted_labels2, fake_predicted_counts2 = discriminator2(fake_images) # discriminator(input = fake images)
        fake_predicted_labels3, fake_predicted_counts3 = discriminator3(fake_images) # discriminator(input = fake images)

        fake_predicted_labels = (fake_predicted_labels1 + fake_predicted_labels2 + fake_predicted_labels3)/3
        generator_density_loss = fake_predicted_labels.sum(1).sum(1).mean()
        fake_predicted_counts = (fake_predicted_counts1 + fake_predicted_counts2 + fake_predicted_counts3)/3
        generator_count_loss = fake_predicted_counts.mean()
        generator_loss = (generator_count_loss + (generator_density_loss * 10)).neg()
############# Generator update ###############
        if step % 5 == 0:
            generator_loss.backward()
            generator_optimizer.step()
##################################### Tensorboard X ##############################################
        running_scalars['Loss'] += loss1.data[0]
        running_scalars['Loss'] += loss2.data[0]
        running_scalars['Loss'] += loss3.data[0]

        running_scalars['Count Loss1'] += count_loss1.data[0]
        running_scalars['Density Loss1'] += density_loss1.data[0]

        running_scalars['Count Loss2'] += count_loss2.data[0]
        running_scalars['Density Loss2'] += density_loss2.data[0]

        running_scalars['Count Loss3'] += count_loss3.data[0]
        running_scalars['Density Loss3'] += density_loss3.data[0]
        
        running_scalars['Fake Discriminator Loss 1'] += fake_discriminator_loss1.data[0]
        
        running_scalars['Fake Discriminator Loss 2'] += fake_discriminator_loss2.data[0]
        
        running_scalars['Fake Discriminator Loss 2'] += fake_discriminator_loss3.data[0]
        
        running_scalars['Generator Loss'] += generator_loss.data[0]


######################################## TensorboardX Comparision of images #################################
        running_example_count += images.size()[0]
        if step % settings.summary_step_period == 0 and step != 0:
            comparison_image1 = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),cpu(predicted_labels1))
            comparison_image2 = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),cpu(predicted_labels2))
            comparison_image3 = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),cpu(predicted_labels3))

            summary_writer.add_image('Comparison1', comparison_image1, global_step=step)
            summary_writer.add_image('Comparison2', comparison_image2, global_step=step)
            summary_writer.add_image('Comparison3', comparison_image3, global_step=step)
            
            fake_images_image = torchvision.utils.make_grid(fake_images.data[:9], nrow=3)
            summary_writer.add_image('Fake', fake_images_image, global_step=step)

##################################### TensorboardX Scalars #########################################
            mean_loss = running_scalars['Loss'] / running_example_count
            print('[Epoch: {}, Step: {}] Loss: {:g}'.format(epoch, step, mean_loss))

            for name, running_scalar in running_scalars.items():
                mean_scalar = running_scalar / running_example_count
                summary_writer.add_scalar(name, mean_scalar, global_step=step)
                running_scalars[name] = 0
            running_example_count = 0
            
################################## Validation Code begins####################################################
            for validation_examples in validation_dataset_loader:
                images, labels, _ = validation_examples
                images, labels = Variable(gpu(images)), Variable(gpu(labels))
                predicted_labels, predicted_counts = discriminator1(images) # discriminator (images)
                
                density_loss = torch.abs(predicted_labels - labels).pow(settings.loss_order).sum(1).sum(1).mean()
                count_loss = torch.abs(predicted_counts - labels.sum(1).sum(1)).pow(settings.loss_order).mean()
                count_mae = torch.abs(predicted_counts - labels.sum(1).sum(1)).mean()
############################### Tensorboard X scalars #################################################
                validation_running_scalars['Density Loss'] += density_loss.data[0]
                validation_running_scalars['Count Loss'] += count_loss.data[0]
                validation_running_scalars['Count MAE'] += count_mae.data[0]
############################### Tensorboard comparision image ################################
            comparison_image = viewer.create_crowd_images_comparison_grid(cpu(images), cpu(labels),cpu(predicted_labels))
            validation_summary_writer.add_image('Comparison', comparison_image, global_step=step)

            for name, running_scalar in validation_running_scalars.items():
                mean_scalar = running_scalar / len(validation_dataset)
                validation_summary_writer.add_scalar(name, mean_scalar, global_step=step)
                validation_running_scalars[name] = 0
############################### 
        step += 1
    epoch += 1
    discriminator_scheduler1.step(epoch)
    generator_scheduler.step(epoch)
#################################   Saving discriminator,generator  after training #############################
    if epoch != 0 and epoch % settings.save_epoch_period == 0:

        save_trainer(trial_directory, discriminator1, discriminator_optimizer1, epoch, step, prefix='discriminator1')
        save_trainer(trial_directory, discriminator2, discriminator_optimizer2, epoch, step, prefix='discriminator2')
        save_trainer(trial_directory, discriminator3, discriminator_optimizer3, epoch, step, prefix='discriminator3')
        
        save_trainer(trial_directory, generator, generator_optimizer, epoch, step, prefix='generator')

save_trainer(trial_directory, discriminator1, discriminator_optimizer1, epoch, step, prefix='discriminator')
save_trainer(trial_directory, discriminator2, discriminator_optimizer2, epoch, step, prefix='discriminator')
save_trainer(trial_directory, discriminator3, discriminator_optimizer3, epoch, step, prefix='discriminator')

save_trainer(trial_directory, generator, generator_optimizer, epoch, step, prefix='generator')
print('Finished Training')
