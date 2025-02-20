use actix_cors::Cors;
use actix_multipart::Field;
use actix_multipart::Multipart;
use actix_web::http::header::ContentDisposition;
use actix_web::{web, web::Data, App, HttpResponse, HttpServer, Responder};
use futures_util::stream::StreamExt as _;
use serde_json::json;
use std::fs;
use std::fs::File;
use std::io::Error;
use std::io::Result as ioResult;
use std::path::Path;
use std::process::Command;
use std::process::Output;
use std::sync::Mutex;
use tracing::{error, info, trace, warn};

use tracing_subscriber::EnvFilter;
struct AppState {
    cache_dir: String,
}

async fn upscale_image_post(data: Data<Mutex<AppState>>, mut payload: Multipart) -> impl Responder {
    info!("Received request to upscale image");
    let cache_dir: String = data.lock().unwrap().cache_dir.clone();
    let mut image_name: String = String::new();
    let mut original_path: String = String::new();
    let mut upscaled_path: String = String::new();

    while let Some(item) = payload.next().await {
        let mut field: Field = item.unwrap();
        let content_disposition: &ContentDisposition = field.content_disposition().unwrap();
        image_name = content_disposition.get_filename().unwrap().to_string();
        info!("Processing file: {}", image_name);
        original_path = format!("{}/{}", cache_dir, image_name);
        upscaled_path = format!("{}/upscaled_{}", cache_dir, image_name);

        let file_result: Result<fs::File, Error> = web::block({
            let original_path: String = original_path.clone();
            move || File::create(&original_path)
        })
        .await
        .unwrap();

        let mut f: fs::File = file_result.unwrap();
        info!("Saving file to: {}", original_path);
        while let Some(chunk) = field.next().await {
            let data: web::Bytes = chunk.unwrap();
            web::block({
                let mut f = f.try_clone().expect("Failed to clone file handle");
                move || {
                    std::io::Write::write_all(&mut f, &data)
                        .map(|_| f)
                        .expect("Failed to write data to file")
                }
            })
            .await
            .unwrap();
        }
    }

    info!("Starting upscaling process for: {}", original_path);
    let command_path = "./realesrgan-ncnn-ubuntu/realesrgan-ncnn-vulkan";
    let args = [
        "-i",
        &original_path,
        "-o",
        &upscaled_path,
        "-g",
        "0",
        "-s",
        "2",
    ];
    let mut command = {
        let mut cmd = Command::new(command_path);
        cmd.args(&args);
        cmd
    };
    trace!("Running command: {:?}", command);

    let output: Result<Output, Error> = command.output();

    match output {
        Ok(_) => {
            if Path::new(&upscaled_path).exists() {
                info!("Upscaling successful, file saved to: {}", upscaled_path);
                HttpResponse::Ok().body(format!("Upscaled image available at: {}", upscaled_path))
            } else {
                error!("Upscaling failed, upscaled file not found");
                HttpResponse::InternalServerError().body("Failed to upscale image")
            }
        }
        Err(e) => {
            error!("Error running upscaling process: {:?}", e);
            if let Some(8) = e.raw_os_error() {
                error!("Exec format error: This might be due to an incompatible binary format.");
            }
            HttpResponse::InternalServerError().body("Error running upscaling process")
        }
    }
}

async fn ping() -> impl Responder {
    info!("Received ping request");
    HttpResponse::Ok().json(json!({"status": "healthy", "data": "pong"}))
}

#[actix_web::main]
async fn main() -> ioResult<()> {
    info!("Starting server");
    init_tracing();

    let cache_dir: &str = "./cache";
    fs::create_dir_all(cache_dir)?;
    info!("Cache directory created at: {}", cache_dir);

    let app_state: Data<Mutex<AppState>> = Data::new(Mutex::new(AppState {
        cache_dir: cache_dir.to_string(),
    }));

    HttpServer::new(move || {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .route("/upscale", web::post().to(upscale_image_post))
            .route("/ping", web::get().to(ping))
    })
    .bind("127.0.0.1:3443")?
    .run()
    .await
}

/// ## Initialize Tracing
///
/// This function sets up the tracing subscriber for logging and monitoring.
///
/// ### Example
///
/// ```
/// init_tracing();
/// ```
fn init_tracing() {
    let filter: EnvFilter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::fmt().with_env_filter(filter).init()
}
