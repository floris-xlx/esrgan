use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::Mutex;

struct AppState {
    cache_dir: String,
}

async fn upscale_image(
    data: web::Data<Mutex<AppState>>,
    image_path: web::Path<String>,
) -> impl Responder {
    let image_name: String = image_path.into_inner();
    let original_path: String = format!("{}/{}", data.lock().unwrap().cache_dir, image_name);
    let upscaled_path: String =
        format!("{}/upscaled_{}", data.lock().unwrap().cache_dir, image_name);

    // Check if the original image exists
    if !Path::new(&original_path).exists() {
        return HttpResponse::NotFound().body("Original image not found");
    }

    // Run the upscaling process
    let output: Result<std::process::Output, std::io::Error> =
        Command::new("./realesrgan-ncnn-ubuntu/realesrgan-ncnn-vulkan")
            .args(&[
                "-i",
                &original_path,
                "-o",
                &upscaled_path,
                "-g",
                "0",
                "-s",
                "2",
            ])
            .output();

    match output {
        Ok(_) => {
            if Path::new(&upscaled_path).exists() {
                HttpResponse::Ok().body(format!("Upscaled image available at: {}", upscaled_path))
            } else {
                HttpResponse::InternalServerError().body("Failed to upscale image")
            }
        }
        Err(_) => HttpResponse::InternalServerError().body("Error running upscaling process"),
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let cache_dir: &str = "./cache";
    fs::create_dir_all(cache_dir)?;

    let app_state = web::Data::new(Mutex::new(AppState {
        cache_dir: cache_dir.to_string(),
    }));

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .route("/upscale/{image_name}", web::get().to(upscale_image))
    })
    .bind("127.0.0.1:3443")?
    .run()
    .await
}
