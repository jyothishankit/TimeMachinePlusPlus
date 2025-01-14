import torch
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
class Model(torch.nn.Module):
    def __init__(self,configs):
        super(Model, self).__init__()
        self.configs=configs
        if self.configs.revin==1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.lin1=torch.nn.Linear(self.configs.seq_len,512)
        self.dropout1=torch.nn.Dropout(self.configs.dropout)

        self.lin2=torch.nn.Linear(512,256)
        self.dropout2=torch.nn.Dropout(self.configs.dropout)
        if self.configs.ch_ind==1:
            self.d_model_param1=1
            self.d_model_param2=1
            self.d_model_param3=1
            self.d_model_param4=1

        else:
            self.d_model_param1=256
            self.d_model_param2=512
            self.d_model_param3=128
            self.d_model_param4=16

        self.mamba1=Mamba(d_model=self.d_model_param1,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba2=Mamba(d_model=256,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba3=Mamba(d_model=512,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba4=Mamba(d_model=self.d_model_param2,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)

        self.lin3=torch.nn.Linear(256,512)
        self.lin4=torch.nn.Linear(4*512,self.configs.pred_len)



        self.lin5=torch.nn.Linear(256,128)
        self.dropout3=torch.nn.Dropout(self.configs.dropout)

        self.lin6=torch.nn.Linear(128,16)
        self.dropout4=torch.nn.Dropout(self.configs.dropout)

        self.mamba5=Mamba(d_model=self.d_model_param3,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba6=Mamba(d_model=128,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact) 
        self.mamba7=Mamba(d_model=self.d_model_param4,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)
        self.mamba8=Mamba(d_model=16,d_state=self.configs.d_state,d_conv=self.configs.dconv,expand=self.configs.e_fact)

        self.lin7=torch.nn.Linear(128,512)
        self.lin8=torch.nn.Linear(16,512)





    def forward(self, x):
         if self.configs.revin==1:
             x=self.revin_layer(x,'norm')
         else:
             means = x.mean(1, keepdim=True).detach()
             x = x - means
             stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
             x /= stdev
         
         x=torch.permute(x,(0,2,1))
         if self.configs.ch_ind==1:
             x=torch.reshape(x,(x.shape[0]*x.shape[1],1,x.shape[2]))

         x=self.lin1(x)
         x_res1=x
         x=self.dropout1(x)
         x3=self.mamba3(x)
         if self.configs.ch_ind==1:
             x4=torch.permute(x,(0,2,1))
         else:
             x4=x
         x4=self.mamba4(x4)
         if self.configs.ch_ind==1:
             x4=torch.permute(x4,(0,2,1))

        
         x4=x4+x3
         

         x=self.lin2(x)
         x_res2=x
         x=self.dropout2(x)
         x_mamba_block_3_input=x   

         if self.configs.ch_ind==1:
             x1=torch.permute(x,(0,2,1))
         else:
             x1=x      

         x1=self.mamba1(x1)    

         if self.configs.ch_ind==1:
             x1=torch.permute(x1,(0,2,1))
  
         x2=self.mamba2(x)

         if self.configs.residual==1:
             x=x1+x_res2+x2
         else:
             x=x1+x2
         
         x=self.lin3(x)
         if self.configs.residual==1:
             x=x+x_res1

         #Mamba block 3 output
         x_mamba_block_3_output = 0
         x5=self.lin5(x_mamba_block_3_input)
         x_res3=x5
         x5=self.dropout3(x5)
         x6_input=x5
         x_mamba_block_4_input=x5
         if self.configs.ch_ind==1:
             x5=torch.permute(x5,(0,2,1))
         else:
             x5=x5
         
         x5=self.mamba5(x5)

         if self.configs.ch_ind==1:
             x5=torch.permute(x5,(0,2,1)) 
         
         x6=self.mamba6(x6_input)

         if self.configs.residual==1:
             x_mamba_block_3_output=x5+x_res3+x6
         else:
             x_mamba_block_3_output=x5+x6

         x_mamba_block_3_output=self.lin7(x_mamba_block_3_output)
         #  if self.configs.residual==1:
         #     x=x+x_mamba_block_3_output

         #Mamba block 4 output
         x_mamba_block_4_output = 0
         x7=self.lin6(x_mamba_block_4_input)
         x_res4=x7
         x7=self.dropout4(x7)
         x8_input=x7
         if self.configs.ch_ind==1:
             x7=torch.permute(x7,(0,2,1))
         else:
             x7=x7
         x7=self.mamba7(x7)
         if self.configs.ch_ind==1:
             x7=torch.permute(x7,(0,2,1))
         
         x8=self.mamba8(x8_input)

         if self.configs.residual==1:
             x_mamba_block_4_output=x7+x_res4+x8
         else:
             x_mamba_block_4_output=x7+x8

         x_mamba_block_4_output=self.lin8(x_mamba_block_4_output)
         #  if self.configs.residual==1:
         #     x=x+x_mamba_block_4_output
             
         x=torch.cat([x,x4],dim=2)
         x=torch.cat([x,x_mamba_block_3_output],dim=2)
         x=torch.cat([x,x_mamba_block_4_output],dim=2)

         x=self.lin4(x) 
         if self.configs.ch_ind==1:
             x=torch.reshape(x,(-1,self.configs.enc_in,self.configs.pred_len))
         
         x=torch.permute(x,(0,2,1))
         if self.configs.revin==1:
             x=self.revin_layer(x,'denorm')
         else:
             x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
             x = x + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
        

         return x
