function sul = compute_sul(Sll, Qll, Socean, Qtide, Ck, Ly, utidef, H, Lx, Qriver)
    Lx = Lx*1000;
    sul_qul = Sll*Qll + Socean*Qtide + Ck*Ly*utidef*H*Ly*(Socean/Lx);
    sul = sul_qul / compute_qul(Qriver,Qll,Qtide);
end