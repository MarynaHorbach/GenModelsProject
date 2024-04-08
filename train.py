def train(args, train_loader, encoder_lmk, encoder_target, generator, decoder, discriminator, bald_model, smooth_mask, device):

    train_loader = sample_data(train_loader)
    generator.train()
    encoder_lmk.train()
    encoder_target.train()
    discriminator.train()

    zero_latent = torch.zeros((args.batch,18-args.coarse,512)).to(device).detach()

    trans_256 = transforms.Resize(256)
    trans_1024 = transforms.Resize(1024)






    pbar = range(args.sample_number)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)
    

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.sample_number:
            print("Done!")
            break

        time0 = time.time()

        s_img,s_code,s_map,s_lmk,t_img,t_code,t_map,t_lmk,t_mask,s_index, t_index = next(train_loader) #256;1024;...
        time1 = time.time()
        s_img = s_img.to(device)
        s_map = s_map.to(device).transpose(1,3).float()#[:,33:]
        t_img = t_img.to(device)
        t_map = t_map.to(device).transpose(1,3).float()#[:,33:]
        t_lmk = t_lmk.to(device)
        t_mask = t_mask.to(device)

        s_frame_code = s_code.to(device)
        t_frame_code = t_code.to(device)



        input_map = torch.cat([s_map,t_map],dim=1)
        t_mask = t_mask.unsqueeze_(1).float()

        t_lmk_code = encoder_lmk(input_map) 


        t_lmk_code = torch.cat([t_lmk_code,zero_latent],dim=1)
        fusion_code = s_frame_code + t_lmk_code


        fusion_code = torch.cat([fusion_code[:,:18-args.coarse],t_frame_code[:,18-args.coarse:]],dim=1)
        fusion_code = bald_model(fusion_code.view(fusion_code.size(0), -1), 2)
        fusion_code = fusion_code.view(t_frame_code.size())


        source_feas = generator([fusion_code], input_is_latent=True, randomize_noise=False)
        target_feas = encoder_target(t_img)

        blend_img = decoder(source_feas,target_feas,t_mask)



        name = str(int(s_index[0]))+'_'+str(int(t_index[0]))


        with torch.no_grad():
            sample = torch.cat([s_img.detach(), t_img.detach()])
            sample = torch.cat([sample, blend_img.detach()])
            t_mask = torch.stack([t_mask,t_mask,t_mask],dim=1).squeeze(2)
            sample = torch.cat([sample, t_mask.detach()])

            utils.save_image(
                blend_img,
                _dirs[0]+"/"+name+".png",
                nrow=int(args.batch),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                sample,
                _dirs[1]+"/"+name+".png",
                nrow=int(args.batch),
                normalize=True,
                range=(-1, 1),
            )



    # https://github.com/lucidrains/stylegan2-pytorch/blob/master/stylegan2_pytorch/stylegan2_pytorch.py#L501
            

    # discriminator train
    generated_images = blend_img.clone()
            
    fake_output, fake_q_loss = discriminator(generated_images.clone().detach(), detach = True, **aug_kwargs)

    real_output, real_q_loss = discriminator(image_batch, **aug_kwargs)

    real_output_loss = real_output
    fake_output_loss = fake_output

    if self.rel_disc_loss:
        real_output_loss = real_output_loss - fake_output.mean()
        fake_output_loss = fake_output_loss - real_output.mean()

    divergence = D_loss_fn(real_output_loss, fake_output_loss)
    disc_loss = divergence


    disc_loss = disc_loss / self.gradient_accumulate_every
    disc_loss.register_hook(raise_if_nan)
    backwards(disc_loss, self.GAN.D_opt, loss_id = 1)

    total_disc_loss += divergence.detach().item() / self.gradient_accumulate_every



    # generator train