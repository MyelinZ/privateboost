use crate::cfg;
use pbr_server::shareholder::serve;

#[tokio::test]
async fn zero_x_coord_rejected() {
    let res = serve(cfg(0)).await;
    assert!(res.is_err());
}

#[tokio::test]
async fn non_loopback_internal_listen_rejected() {
    let mut c = cfg(1);
    c.internal_listen = "0.0.0.0:0".parse().unwrap();
    let res = serve(c).await;
    assert!(res.is_err());
}
