; ModuleID = 'Test.c'
source_filename = "Test.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

@.str = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @init(ptr noundef %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store i64 %3, ptr %8, align 8
  store i64 0, ptr %9, align 8
  br label %10

10:                                               ; preds = %38, %4
  %11 = load i64, ptr %9, align 8
  %12 = load i64, ptr %8, align 8
  %13 = icmp ult i64 %11, %12
  br i1 %13, label %14, label %41

14:                                               ; preds = %10
  %15 = load i64, ptr %8, align 8
  %16 = load i64, ptr %9, align 8
  %17 = sub i64 %15, %16
  %18 = uitofp i64 %17 to float
  %19 = load ptr, ptr %5, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds float, ptr %19, i64 %20
  store float %18, ptr %21, align 4
  %22 = load i64, ptr %8, align 8
  %23 = load i64, ptr %8, align 8
  %24 = add i64 %22, %23
  %25 = load i64, ptr %9, align 8
  %26 = add i64 %24, %25
  %27 = uitofp i64 %26 to float
  %28 = load ptr, ptr %6, align 8
  %29 = load i64, ptr %9, align 8
  %30 = getelementptr inbounds float, ptr %28, i64 %29
  store float %27, ptr %30, align 4
  %31 = load i64, ptr %9, align 8
  %32 = load i64, ptr %8, align 8
  %33 = add i64 %31, %32
  %34 = uitofp i64 %33 to float
  %35 = load ptr, ptr %7, align 8
  %36 = load i64, ptr %9, align 8
  %37 = getelementptr inbounds float, ptr %35, i64 %36
  store float %34, ptr %37, align 4
  br label %38

38:                                               ; preds = %14
  %39 = load i64, ptr %9, align 8
  %40 = add i64 %39, 1
  store i64 %40, ptr %9, align 8
  br label %10, !llvm.loop !6

41:                                               ; preds = %10
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @VectorAdd(ptr noundef %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store i64 %3, ptr %8, align 8
  store i64 0, ptr %9, align 8
  br label %10

10:                                               ; preds = %32, %4
  %11 = load i64, ptr %9, align 8
  %12 = load i64, ptr %8, align 8
  %13 = icmp ult i64 %11, %12
  br i1 %13, label %14, label %35

14:                                               ; preds = %10
  %15 = load ptr, ptr %5, align 8
  %16 = load i64, ptr %9, align 8
  %17 = getelementptr inbounds float, ptr %15, i64 %16
  %18 = load float, ptr %17, align 4
  %19 = load ptr, ptr %6, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds float, ptr %19, i64 %20
  %22 = load float, ptr %21, align 4
  %23 = fadd float %18, %22
  %24 = load ptr, ptr %5, align 8
  %25 = load i64, ptr %9, align 8
  %26 = getelementptr inbounds float, ptr %24, i64 %25
  %27 = load float, ptr %26, align 4
  %28 = fdiv float %23, %27
  %29 = load ptr, ptr %7, align 8
  %30 = load i64, ptr %9, align 8
  %31 = getelementptr inbounds float, ptr %29, i64 %30
  store float %28, ptr %31, align 4
  br label %32

32:                                               ; preds = %14
  %33 = load i64, ptr %9, align 8
  %34 = add i64 %33, 1
  store i64 %34, ptr %9, align 8
  br label %10, !llvm.loop !8

35:                                               ; preds = %10
  ret void
}

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define void @FuseAddMul(ptr noundef %0, ptr noundef %1, ptr noundef %2, i64 noundef %3) #0 {
  %5 = alloca ptr, align 8
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  store ptr %0, ptr %5, align 8
  store ptr %1, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store i64 %3, ptr %8, align 8
  store i64 0, ptr %9, align 8
  br label %10

10:                                               ; preds = %31, %4
  %11 = load i64, ptr %9, align 8
  %12 = load i64, ptr %8, align 8
  %13 = icmp ult i64 %11, %12
  br i1 %13, label %14, label %34

14:                                               ; preds = %10
  %15 = load ptr, ptr %5, align 8
  %16 = load i64, ptr %9, align 8
  %17 = getelementptr inbounds float, ptr %15, i64 %16
  %18 = load float, ptr %17, align 4
  %19 = load ptr, ptr %6, align 8
  %20 = load i64, ptr %9, align 8
  %21 = getelementptr inbounds float, ptr %19, i64 %20
  %22 = load float, ptr %21, align 4
  %23 = load ptr, ptr %7, align 8
  %24 = load i64, ptr %9, align 8
  %25 = getelementptr inbounds float, ptr %23, i64 %24
  %26 = load float, ptr %25, align 4
  %27 = call float @llvm.fmuladd.f32(float %18, float %22, float %26)
  %28 = load ptr, ptr %7, align 8
  %29 = load i64, ptr %9, align 8
  %30 = getelementptr inbounds float, ptr %28, i64 %29
  store float %27, ptr %30, align 4
  br label %31

31:                                               ; preds = %14
  %32 = load i64, ptr %9, align 8
  %33 = add i64 %32, 1
  store i64 %33, ptr %9, align 8
  br label %10, !llvm.loop !9

34:                                               ; preds = %10
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare float @llvm.fmuladd.f32(float, float, float) #1

; Function Attrs: noinline nounwind optnone ssp uwtable(sync)
define i32 @main() #0 {
  %1 = alloca i32, align 4
  %2 = alloca [10 x float], align 4
  %3 = alloca [10 x float], align 4
  %4 = alloca [10 x float], align 4
  %5 = alloca i64, align 8
  store i32 0, ptr %1, align 4
  %6 = getelementptr inbounds [10 x float], ptr %2, i64 0, i64 0
  %7 = getelementptr inbounds [10 x float], ptr %3, i64 0, i64 0
  %8 = getelementptr inbounds [10 x float], ptr %4, i64 0, i64 0
  call void @init(ptr noundef %6, ptr noundef %7, ptr noundef %8, i64 noundef 10)
  %9 = getelementptr inbounds [10 x float], ptr %2, i64 0, i64 0
  %10 = getelementptr inbounds [10 x float], ptr %3, i64 0, i64 0
  %11 = getelementptr inbounds [10 x float], ptr %4, i64 0, i64 0
  call void @VectorAdd(ptr noundef %9, ptr noundef %10, ptr noundef %11, i64 noundef 10)
  %12 = getelementptr inbounds [10 x float], ptr %2, i64 0, i64 0
  %13 = getelementptr inbounds [10 x float], ptr %3, i64 0, i64 0
  %14 = getelementptr inbounds [10 x float], ptr %4, i64 0, i64 0
  call void @FuseAddMul(ptr noundef %12, ptr noundef %13, ptr noundef %14, i64 noundef 10)
  store i64 0, ptr %5, align 8
  br label %15

15:                                               ; preds = %24, %0
  %16 = load i64, ptr %5, align 8
  %17 = icmp ult i64 %16, 10
  br i1 %17, label %18, label %27

18:                                               ; preds = %15
  %19 = load i64, ptr %5, align 8
  %20 = getelementptr inbounds [10 x float], ptr %4, i64 0, i64 %19
  %21 = load float, ptr %20, align 4
  %22 = fpext float %21 to double
  %23 = call i32 (ptr, ...) @printf(ptr noundef @.str, double noundef %22)
  br label %24

24:                                               ; preds = %18
  %25 = load i64, ptr %5, align 8
  %26 = add i64 %25, 1
  store i64 %26, ptr %5, align 8
  br label %15, !llvm.loop !10

27:                                               ; preds = %15
  %28 = load i32, ptr %1, align 4
  ret i32 %28
}

declare i32 @printf(ptr noundef, ...) #2

attributes #0 = { noinline nounwind optnone ssp uwtable(sync) "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }
attributes #1 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "probe-stack"="__chkstk_darwin" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+crc,+crypto,+dotprod,+fp-armv8,+fp16fml,+fullfp16,+lse,+neon,+ras,+rcpc,+rdm,+sha2,+sha3,+sm4,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+zcm,+zcz" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 14, i32 0]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Apple clang version 15.0.0 (clang-1500.0.40.1)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
