import struct,re
W,H=512,1024
FONTS=['bigFont','smallFont','boldFont','normalFont','extraBigFont','objectiveFont']
def parse(zone,e,names=FONTS):
    out={}
    for n in names:
        m=re.search(rb'fonts/'+n.encode()+rb'\x00',zone)
        s=m.start(); hdr=struct.unpack(e+'6i',zone[s-24:s]); g0=m.end()
        cnt=hdr[2]; gl=[]
        for i in range(cnt):
            r=zone[g0+i*24:g0+i*24+24]
            let=struct.unpack(e+'H',r[:2])[0]; x0,y0=struct.unpack('bb',r[2:4]); dx,pw,ph=r[4],r[5],r[6]
            s0,t0,s1,t1=struct.unpack(e+'4f',r[8:24])
            gl.append(dict(let=let,x0=x0,y0=y0,dx=dx,pw=pw,ph=ph,uv=(s0,t0,s1,t1),off=g0+i*24,pad=r[7]))
        out[n]=dict(hdr=hdr,pixh=hdr[1],g0=g0,gl=gl,hoff=s-24)
    return out
def rect(g): 
    s0,t0,s1,t1=g['uv']; return (round(s0*W),round(t0*H),round(s1*W),round(t1*H))
