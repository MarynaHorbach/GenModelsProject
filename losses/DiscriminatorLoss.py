class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, fake_disc_out: Tensor, real_disc_out: Tensor, **kwargs):
        labels = torch.ones_like(fake_disc_out).to(fake_disc_out)
        return self.bce_loss(fake_disc_out, labels) + self.bce_loss(real_disc_out, labels)