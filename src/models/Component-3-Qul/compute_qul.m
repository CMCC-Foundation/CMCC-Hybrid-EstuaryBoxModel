% Function to compute the Component-3 of Hybrid-EBM
function Qul = compute_qul(Qriver, Qll, Qtide)
    Qul = Qriver + Qll + Qtide;
end