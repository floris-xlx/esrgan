use actix_cors::Cors;
use actix_multipart::Field;
use actix_multipart::Multipart;
use actix_web::http::header::ContentDisposition;
use actix_web::{web, web::Data, App, HttpResponse, HttpServer, Responder};
use futures_util::stream::StreamExt as _;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Error;
use std::io::Result as ioResult;
use std::path::Path;
use std::process::Command;
use std::process::Output;
use tokio::sync::Mutex;
use tracing::{error, info, trace, warn};
use uuid::Uuid;
use tokio::sync::MutexGuard;

use tracing_subscriber::EnvFilter;

struct AppState {
    cache_dir: String,
    request_status: Mutex<HashMap<String, String>>,
}

async fn upscale_image_post(data: Data<Mutex<AppState>>, mut payload: Multipart) -> impl Responder {
    info!("Received request to upscale image");
    let cache_dir: String = data.lock().await.cache_dir.clone();
    let request_id: String = Uuid::new_v4().to_string();
    info!("Started computing request ID: {}", request_id);
    let mut image_name: String = String::new();
    let mut original_path: String = String::new();
    let mut upscaled_path: String = String::new();

    let app_state: MutexGuard<'_, AppState> = data.lock().await;
    let mut status_map: MutexGuard<'_, HashMap<String, String>> =
        app_state.request_status.lock().await;
    status_map.insert(request_id.clone(), "Processing".to_string());

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

    use tokio::process::Command;
    use tokio::io::{self, AsyncBufReadExt};
    use std::process::Stdio;

    info!("Starting upscaling process for: {}", original_path);
    let command_path: &str = "./realesrgan-ncnn-ubuntu/realesrgan-ncnn-vulkan";
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
    let mut command: Command = {
        let mut cmd: Command = Command::new(command_path);
        cmd.args(&args);
        cmd.stdout(Stdio::piped());
        cmd
    };
    trace!("Running command: {:?}", command);

    let mut child = command.spawn().expect("Failed to spawn command");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    let reader = io::BufReader::new(stdout);
    let mut lines = reader.lines();

    while let Some(line) = lines.next_line().await.expect("Failed to read line") {
        println!("Upscaling process output: {}", line);
    }

    let output = child.wait_with_output().await;

    let mut data_lock: MutexGuard<'_, AppState> = data.lock().await;
    let mut status_map: MutexGuard<'_, HashMap<String, String>> =
        data_lock.request_status.lock().await;

    let status = match output {
        Ok(_) => {
            if Path::new(&upscaled_path).exists() {
                info!("Upscaling successful, file saved to: {}", upscaled_path);
                status_map.insert(request_id.clone(), "Completed".to_string());
                "Completed"
            } else {
                error!("Upscaling failed, upscaled file not found");
                status_map.insert(request_id.clone(), "Failed".to_string());
                "Failed"
            }
        }
        Err(e) => {
            error!("Error running upscaling process: {:?}", e);
            if let Some(8) = e.raw_os_error() {
                error!("Exec format error: This might be due to an incompatible binary format.");
            }
            status_map.insert(request_id.clone(), "Error".to_string());
            "Error"
        }
    };

    HttpResponse::Ok().json(json!({"status": status, "data": {"request_id": request_id}}))
}

async fn get_status(data: Data<Mutex<AppState>>, request_id: web::Path<String>) -> impl Responder {
    let request_id: String = request_id.into_inner();
    let status_map_lock: MutexGuard<'_, AppState> = data.lock().await;
    let status_map: MutexGuard<'_, HashMap<String, String>> =
        status_map_lock.request_status.lock().await;
    if let Some(status) = status_map.get(&request_id) {
        HttpResponse::Ok().json(json!({"status": status}))
    } else {
        HttpResponse::NotFound().body("Request ID not found")
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
        request_status: Mutex::new(HashMap::new()),
    }));

    HttpServer::new(move || {
        let cors = Cors::permissive();

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .route("/upscale", web::post().to(upscale_image_post))
            .route("/status/{request_id}", web::get().to(get_status))
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
