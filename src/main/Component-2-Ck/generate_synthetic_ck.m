function ck = generate_synthetic_ck(Sul, Qul, Lx_ml, h, Ly, Socean, utide, Qll, Sll)
    Lx_ml = Lx_ml * 1000;
    ck = ((Sul*Qul*Lx_ml)/(h*(Ly^2)*Socean*utide))-((Sll*Qll*Lx_ml)/(h*(Ly^2)*Socean*utide))-(Lx_ml/Ly);
end