claude# Guía de Compilación Protegida

Esta guía explica cómo compilar `ai_assistant` con protecciones anti-ingeniería inversa.

## Índice

1. [Requisitos Previos](#requisitos-previos)
2. [Perfiles de Compilación](#perfiles-de-compilación)
3. [Compilación Protegida](#compilación-protegida)
4. [Verificación de Integridad](#verificación-de-integridad)
5. [Integración en Proyectos Dependientes](#integración-en-proyectos-dependientes)
6. [Protecciones Aplicadas](#protecciones-aplicadas)

---

## Requisitos Previos

### Obligatorios

- **Rust** (stable o nightly)
  ```powershell
  # Verificar instalación
  rustc --version
  cargo --version
  ```

### Opcionales (para máxima protección)

- **UPX** - Compresor de ejecutables
  ```powershell
  # Descargar desde: https://github.com/upx/upx/releases
  # Extraer y añadir al PATH, o colocar en C:\Tools\upx\

  # Verificar
  upx --version
  ```

- **OpenSSL** - Para firma de binarios
  ```powershell
  # Opción 1: Instalar via Chocolatey
  choco install openssl

  # Opción 2: Descargar desde https://slproweb.com/products/Win32OpenSSL.html

  # Verificar
  openssl version
  ```

---

## Perfiles de Compilación

El proyecto incluye varios perfiles de release:

| Perfil | Uso | Tamaño | Velocidad |
|--------|-----|--------|-----------|
| `release` | Producción estándar | Pequeño | Rápida |
| `release-fast` | Rendimiento máximo | Medio | Muy rápida |
| `release-protected` | Anti-ingeniería inversa | Muy pequeño | Rápida |

### Comparación de Optimizaciones

```toml
# release (estándar)
lto = true
codegen-units = 1
opt-level = "z"

# release-protected (máxima protección)
lto = "fat"           # LTO completo, fusiona todo el código
codegen-units = 1     # Un solo bloque de código
opt-level = "z"       # Optimización de tamaño
strip = "symbols"     # Sin símbolos de debug
panic = "abort"       # Sin información de panic
overflow-checks = false
```

---

## Compilación Protegida

### Método 1: Script Automatizado (Recomendado)

```powershell
# Compilación básica
.\scripts\build_protected.ps1

# Sin UPX (si no está instalado)
.\scripts\build_protected.ps1 -SkipUpx

# Con firma de binario
.\scripts\build_protected.ps1 -SignBinary -PrivateKeyPath "keys\private.pem"

# Verbose
.\scripts\build_protected.ps1 -Verbose
```

El script realiza:
1. Compilación con perfil `release-protected`
2. Compresión UPX (si disponible)
3. Generación de hash SHA256
4. Firma opcional del binario
5. Generación de archivo Rust con hash embebido

### Método 2: Compilación Manual

```powershell
# Compilar
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --profile release-protected --features "full,integrity-check"

# El binario estará en:
# target/release-protected/ai_assistant.exe (biblioteca)
# target/release-protected/tu_binario.exe (si es ejecutable)
```

### Método 3: Desde Proyecto Dependiente

En el `Cargo.toml` del proyecto que usa `ai_assistant`:

```toml
[dependencies]
ai_assistant = { path = "../ai_assistant_standalone", features = ["full", "integrity-check"] }

[profile.release-protected]
inherits = "release"
lto = "fat"
codegen-units = 1
opt-level = "z"
strip = "symbols"
panic = "abort"
```

Luego compilar:
```powershell
cargo build --profile release-protected
```

---

## Verificación de Integridad

### Configuración en tu Aplicación

Añade la verificación de integridad al inicio de tu aplicación:

```rust
use ai_assistant::binary_integrity::{IntegrityChecker, startup_integrity_check};

fn main() {
    // Opción 1: Verificación simple (usa hash de variable de entorno)
    startup_integrity_check();

    // Opción 2: Con hash embebido en código
    let checker = IntegrityChecker::new(
        IntegrityConfig::with_hash("abc123...")  // Hash generado por build script
    );
    checker.verify_or_abort();

    // Opción 3: Macro (más conciso)
    ai_assistant::integrity_guard!();

    // Tu código de aplicación...
}
```

### Generar Claves para Firma

```powershell
# Crear directorio de claves
mkdir keys

# Generar clave privada (¡MANTENER SEGURA!)
openssl genrsa -out keys\private.pem 2048

# Generar clave pública (puede incluirse en el binario)
openssl rsa -in keys\private.pem -pubout -out keys\public.pem
```

### Verificar Firma Manualmente

```powershell
# Verificar que el binario no ha sido modificado
openssl dgst -sha256 -verify keys\public.pem -signature target\release-protected\app.sig target\release-protected\app.exe
```

---

## Integración en Proyectos Dependientes

### Opción A: Feature Flag

Los proyectos que dependen de `ai_assistant` pueden activar la verificación:

```toml
[dependencies]
ai_assistant = { version = "0.1", features = ["integrity-check"] }
```

### Opción B: Build Script Compartido

Copia `scripts/build_protected.ps1` a tu proyecto y modifica:

```powershell
# En tu build_protected.ps1
param(
    [string]$BinaryName = "mi_aplicacion",  # Cambia el nombre
    [string]$Features = "full,integrity-check"
)
```

### Opción C: Cargo Make (CI/CD)

Crea un `Makefile.toml`:

```toml
[tasks.build-protected]
script = '''
$env:RUSTFLAGS = "-C target-feature=+crt-static"
cargo build --profile release-protected --features "full,integrity-check"
'''

[tasks.sign]
script = '''
openssl dgst -sha256 -sign keys/private.pem -out target/release-protected/app.sig target/release-protected/app.exe
'''

[tasks.release-protected]
dependencies = ["build-protected", "sign"]
```

---

## Protecciones Aplicadas

### Sin Penalización en Runtime

| Protección | Descripción | Impacto |
|------------|-------------|---------|
| **Fat LTO** | Fusiona todo el código en un bloque | 0% runtime |
| **Symbol Stripping** | Elimina nombres de funciones/variables | 0% runtime |
| **Single Codegen Unit** | Dificulta identificar módulos | 0% runtime |
| **Panic Abort** | Sin mensajes de error detallados | 0% runtime |
| **Size Optimization** | Ofusca patrones de código | 0% runtime |
| **UPX Compression** | Comprime y ofusca estructura | 2-5% startup |
| **Integrity Check** | Verificación SHA256 al inicio | ~10ms startup |

### Lo que NO se incluye (por penalizar runtime)

- Control Flow Guard (1-3% runtime)
- String encryption (variable)
- Code virtualization (10-50x para código protegido)

### Análisis de Tamaño

Ejemplo de reducción de tamaño típica:

```
Original (debug):        ~50 MB
Release estándar:        ~8 MB
Release protected:       ~5 MB
Release protected + UPX: ~2 MB
```

---

## Troubleshooting

### Error: UPX no encontrado

```
[!] UPX not found in PATH, skipping compression
```

**Solución:** Instala UPX o usa `-SkipUpx`:
```powershell
.\scripts\build_protected.ps1 -SkipUpx
```

### Error: OpenSSL no encontrado

```
[!] OpenSSL not found, skipping signing
```

**Solución:** Instala OpenSSL o omite la firma (no usar `-SignBinary`).

### Error: Compilación muy lenta

El perfil `release-protected` usa Fat LTO, que es más lento. Esto es normal:
- Primera compilación: 5-15 minutos
- Compilaciones incrementales: 2-5 minutos

### Error: Verificación de integridad falla

1. Asegúrate de que el hash en código corresponde al binario compilado
2. El hash debe regenerarse después de cada compilación
3. Usa el script `build_protected.ps1` que genera el hash automáticamente

---

## Notas de Seguridad

1. **La clave privada nunca debe incluirse en el código fuente**
2. **La clave privada debe mantenerse segura** (servidor de build, HSM, etc.)
3. **La verificación de integridad puede ser parcheada** por atacantes avanzados
4. Para máxima seguridad, considera:
   - Verificación externa (servidor de licencias)
   - Certificados de firma de código (Authenticode)
   - Hardware security modules (TPM)

---

## Referencias

- [Rust Release Profiles](https://doc.rust-lang.org/cargo/reference/profiles.html)
- [UPX Documentation](https://upx.github.io/)
- [Code Signing Best Practices](https://docs.microsoft.com/en-us/windows/win32/seccrypto/cryptography-tools)
